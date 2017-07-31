# Standard library
from os.path import abspath, expanduser, join

# Third-party
from astropy.table import Table
import numpy as np

# Project
from ..config import TWOFACE_CACHE_PATH
from ..util import Timer
from ..log import log as logger
from .connect import db_connect, Base
from .model import (AllStar, AllVisit, AllVisitToAllStar, Status, RedClump,
                    StarResult)

__all__ = ['initialize_db', 'load_red_clump']

def initialize_db(allVisit_file, allStar_file, database_file,
                  drop_all=False, overwrite=False, batch_size=4096):
    """Initialize the database given FITS filenames for the APOGEE data.

    Parameters
    ----------
    allVisit_file : str
        Full path to APOGEE allVisit file.
    allStar_file : str
        Full path to APOGEE allStar file.
    database_file : str
        Filename (not path) of database file in cache path.
    drop_all : bool (optional)
        Drop all existing tables and re-create the database.
    overwrite : bool (optional)
        Overwrite any data already loaded into the database.
    batch_size : int (optional)
        How many rows to create before committing.
    """

    database_path = join(TWOFACE_CACHE_PATH, database_file)

    norm = lambda x: abspath(expanduser(x))
    allvisit_tbl = Table.read(norm(allVisit_file), format='fits', hdu=1)
    allvisit_tbl = allvisit_tbl[np.isfinite(allvisit_tbl['VHELIO'])]
    allstar_tbl = Table.read(norm(allStar_file), format='fits', hdu=1)

    Session, engine = db_connect(database_path)
    logger.debug("Connected to database at '{}'".format(database_path))

    if drop_all:
        # this is the magic that creates the tables based on the definitions in
        # twoface/db/model.py
        Base.metadata.drop_all()
        Base.metadata.create_all()

    session = Session()

    logger.debug("Loading allStar, allVisit tables...")

    # Figure out what data we need to pull out of the FITS files based on what
    # columns exist in the (empty) database
    allstar_colnames = [str(x).split('.')[1].upper()
                        for x in AllStar.__table__.columns]
    i = allstar_colnames.index('ID')
    allstar_colnames.pop(i)

    allvisit_colnames = [str(x).split('.')[1].upper()
                         for x in AllVisit.__table__.columns]
    i = allvisit_colnames.index('ID')
    allvisit_colnames.pop(i)

    # What APOGEE IDs are already loaded?
    ap_ids = [x[0] for x in session.query(AllStar.apstar_id).all()]
    logger.debug("{0} stars already loaded".format(len(ap_ids)))

    stars = []
    all_visits = dict()
    with Timer() as t:
        for i,row in enumerate(allstar_tbl): # Load every star
            row_data = dict([(k.lower(), row[k]) for k in allstar_colnames])

            # If this APOGEE ID is already in the database and we are
            # overwriting data, delete that row
            if row_data['apstar_id'] in ap_ids:
                q = session.query(AllStar).filter(
                    AllStar.apstar_id == row_data['apstar_id'])
                star = q.one()

                if overwrite:
                    visits = session.query(AllVisit).join(AllVisitToAllStar,
                                                          AllStar)\
                        .filter(AllStar.apstar_id == star.apstar_id).all()
                    session.delete(star)
                    for v in visits:
                        session.delete(v)
                    session.commit()

                    star = AllStar(**row_data)
                    stars.append(star)

                    logger.log(1, 'Overwriting star {0} in database'
                                  .format(star))

                else:
                    logger.log(1, 'Loaded star {0} from database'.format(star))

            else:
                star = AllStar(**row_data)
                stars.append(star)
                logger.log(1, 'Adding star {0} to database'.format(star))

            if not star.visits or (star.visits and overwrite):
                if star.apogee_id not in all_visits:
                    visits = []
                    rows = allvisit_tbl[allvisit_tbl['APOGEE_ID']==row['APOGEE_ID']]
                    for j,visit_row in enumerate(rows):
                        _data = dict([(k.lower(), visit_row[k])
                                      for k in allvisit_colnames])
                        visits.append(AllVisit(**_data))
                    all_visits[star.apogee_id] = visits

                else:
                    visits = all_visits[star.apogee_id]

                star.visits = visits

            if i % batch_size == 0 and i > 0:
                session.add_all(stars)
                session.add_all([item for sublist in all_visits.values()
                                 for item in sublist])
                session.commit()
                logger.debug("Loaded batch {} ({:.2f} seconds)"
                             .format(i*batch_size, t.elapsed()))
                t.reset()

                all_visits = dict()
                stars = []

    if len(stars) > 0:
        session.add_all(stars)
        session.add_all([item for sublist in all_visits.values()
                         for item in sublist])
        session.commit()

    # Load the status table
    if session.query(Status).count() == 0:
        logger.debug("Populating Status table...")
        statuses = list()
        statuses.append(Status(id=0, message='untouched'))
        statuses.append(Status(id=1, message='needs more prior samples'))
        statuses.append(Status(id=2, message='needs mcmc'))
        statuses.append(Status(id=3, message='error'))
        statuses.append(Status(id=4, message='completed'))

        session.add_all(statuses)
        session.commit()
        logger.debug("...done")

    session.close()

def load_red_clump(filename, database_file, overwrite=False, batch_size=4096):
    """Load the red clump catalog stars into the database.

    Information about this catalog can be found here:
    http://www.sdss.org/dr13/data_access/vac/

    Parameters
    ----------
    filename : str
        Full path to APOGEE red clump FITS file.
    database_file : str
        Filename (not path) of database file in cache path.
    overwrite : bool (optional)
        Overwrite any data already loaded into the database.
    batch_size : int (optional)
        How many rows to create before committing.
    """

    database_path = join(TWOFACE_CACHE_PATH, database_file)

    norm = lambda x: abspath(expanduser(x))
    rc_tbl = Table.read(norm(filename), format='fits', hdu=1)

    Session, engine = db_connect(database_path)
    logger.debug("Connected to database at '{}'".format(database_path))
    session = Session()

    # What columns do we load?
    rc_colnames = [str(x).split('.')[1].upper()
                   for x in RedClump.__table__.columns]
    for name in ['ID', 'ALLSTAR_ID']:
        i = rc_colnames.index(name)
        rc_colnames.pop(i)

    # What APOGEE IDs are already loaded as RC stars?
    rc_ap_ids = session.query(AllStar.apstar_id).join(RedClump).all()
    rc_ap_ids = [x[0] for x in rc_ap_ids]

    rcstars = []
    with Timer() as t:
        for i,row in enumerate(rc_tbl):
            # Only data for columns that exist in the table
            row_data = dict([(k.lower(), row[k]) for k in rc_colnames])

            # Retrieve the parent AllStar record
            try:
                star = session.query(AllStar).filter(
                    AllStar.apstar_id == row['APSTAR_ID']).one()
            except:
                logger.debug('Red clump star not found in AllStar - skipping')
                continue

            if row['APSTAR_ID'] in rc_ap_ids:
                q = session.query(RedClump).join(AllStar).filter(
                    AllStar.apstar_id == row['APSTAR_ID'])

                if overwrite:
                    q.delete()
                    session.commit()

                    rc = RedClump(**row_data)
                    rc.star = star
                    rcstars.append(rc)

                    logger.log(1, 'Overwriting rc {0} in database'
                                  .format(rc))

                else:
                    rc = q.one()
                    logger.log(1, 'Loaded rc {0} from database'.format(rc))

            else:
                rc = RedClump(**row_data)
                rc.star = star
                rcstars.append(rc)
                logger.log(1, 'Adding rc {0} to database'.format(rc))

            if i % batch_size == 0 and i > 0:
                session.add_all(rcstars)
                session.commit()
                logger.debug("Loaded rc batch {} ({:.2f} seconds)"
                             .format(i*batch_size, t.elapsed()))
                t.reset()
                rcstars = []

    if len(rcstars) > 0:
        session.add_all(rcstars)
        session.commit()

    logger.debug("tables loaded in {:.2f} seconds".format(t.elapsed()))

    session.close()