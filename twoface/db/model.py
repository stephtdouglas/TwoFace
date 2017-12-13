from __future__ import division, print_function

# Standard library
import datetime

# Third-party
import astropy.units as u
from sqlalchemy import Table, Column, types
from sqlalchemy.schema import ForeignKey
from sqlalchemy.orm import relationship, backref

# Project
from ..data import star_to_apogeervdata
from .connect import Base
from .quantity_type import QuantityTypeClassFactory
from . import numpy_adapt # just need to execute code

__all__ = ['AllStar', 'AllVisit', 'RedClump', 'JokerRun', 'StarResult',
           'Status', 'AllVisitToAllStar', 'NessRG']

class Status(Base):
    __tablename__ = 'status'

    id = Column(types.Integer, primary_key=True, autoincrement=False)
    message = Column('message', types.String)

    def __repr__(self):
        return "<Status id={0} message='{1}'>".format(self.id, self.message)

class StarResult(Base):
    __tablename__ = 'starresult'

    id = Column(types.Integer, primary_key=True)
    allstar_id = Column('allstar_id', types.Integer,
                        ForeignKey('allstar.id', ondelete='CASCADE'),
                        index=True)
    jokerrun_id = Column('jokerrun_id', types.Integer,
                         ForeignKey('jokerrun.id'),
                         index=True)
    status_id = Column('status_id', types.Integer,
                       ForeignKey('status.id'),
                       default=0, index=True)

    star = relationship("AllStar")
    jokerrun = relationship("JokerRun",
                            backref=backref("results",
                                            cascade="all, delete-orphan"))
    status = relationship("Status")

    def __repr__(self):
        return ("<JokerResult(apogee_id='{}', joker_run='{}', status='{}')>"
                .format(self.star.apogee_id, self.jokerrun.name,
                        self.status.message))

AllVisitToAllStar = Table('allvisit_to_allstar',
                          Base.metadata,
                          Column('allstar_id', types.Integer,
                                 ForeignKey('allstar.id', ondelete='CASCADE'),
                                 index=True),
                          Column('allvisit_id', types.Integer,
                                 ForeignKey('allvisit.id', ondelete='CASCADE'),
                                 index=True))

class AllStar(Base):
    __tablename__ = 'allstar'

    id = Column(types.Integer, primary_key=True)
    visits = relationship("AllVisit", secondary=AllVisitToAllStar,
                          backref=backref("stars"))

    joker_runs = relationship("JokerRun", secondary="starresult",
                              back_populates="stars")
    results = relationship("StarResult")

    # Columns from the FITS file, auto-generated by
    #       scripts/_generate_model_columns.py
    apstar_id = Column("apstar_id", types.String, index=True)
    target_id = Column("target_id", types.String)
    aspcap_id = Column("aspcap_id", types.String)
    file = Column("file", types.String)
    apogee_id = Column("apogee_id", types.String, index=True)
    telescope = Column("telescope", types.String)
    location_id = Column("location_id", types.SmallInteger)
    field = Column("field", types.String)
    j = Column("j", types.REAL)
    j_err = Column("j_err", types.REAL)
    h = Column("h", types.REAL)
    h_err = Column("h_err", types.REAL)
    k = Column("k", types.REAL)
    k_err = Column("k_err", types.REAL)
    ra = Column("ra", types.REAL)
    dec = Column("dec", types.REAL)
    glon = Column("glon", types.REAL)
    glat = Column("glat", types.REAL)
    apogee_target1 = Column("apogee_target1", types.Integer)
    apogee_target2 = Column("apogee_target2", types.Integer)
    apogee_target3 = Column("apogee_target3", types.Integer)
    targflags = Column("targflags", types.String)
    nvisits = Column("nvisits", types.Integer)
    commiss = Column("commiss", types.SmallInteger)
    snr = Column("snr", types.REAL)
    starflag = Column("starflag", types.Integer)
    starflags = Column("starflags", types.String)
    andflag = Column("andflag", types.Integer)
    andflags = Column("andflags", types.String)
    vhelio_avg = Column("vhelio_avg", types.REAL)
    vscatter = Column("vscatter", types.REAL)
    verr = Column("verr", types.REAL)
    verr_med = Column("verr_med", types.REAL)
    synthvhelio_avg = Column("synthvhelio_avg", types.REAL)
    synthvscatter = Column("synthvscatter", types.REAL)
    synthverr = Column("synthverr", types.REAL)
    synthverr_med = Column("synthverr_med", types.REAL)
    rv_teff = Column("rv_teff", types.REAL)
    rv_logg = Column("rv_logg", types.REAL)
    rv_feh = Column("rv_feh", types.REAL)
    rv_alpha = Column("rv_alpha", types.REAL)
    rv_carb = Column("rv_carb", types.REAL)
    rv_ccfwhm = Column("rv_ccfwhm", types.REAL)
    rv_autofwhm = Column("rv_autofwhm", types.REAL)
    synthscatter = Column("synthscatter", types.REAL)
    # stablerv_chi2 = Column("stablerv_chi2", postgresql.ARRAY(types.REAL))
    # stablerv_rchi2 = Column("stablerv_rchi2", postgresql.ARRAY(types.REAL))
    # chi2_threshold = Column("chi2_threshold", postgresql.ARRAY(types.REAL))
    # stablerv_chi2_prob = Column("stablerv_chi2_prob", postgresql.ARRAY(types.REAL))
    meanfib = Column("meanfib", types.REAL)
    sigfib = Column("sigfib", types.REAL)
    snrev = Column("snrev", types.REAL)
    apstar_version = Column("apstar_version", types.String)
    aspcap_version = Column("aspcap_version", types.String)
    results_version = Column("results_version", types.String)
    extratarg = Column("extratarg", types.SmallInteger)
    min_h = Column("min_h", types.REAL)
    max_h = Column("max_h", types.REAL)
    min_jk = Column("min_jk", types.REAL)
    # param = Column("param", postgresql.ARRAY(types.REAL))
    # fparam = Column("fparam", postgresql.ARRAY(types.REAL))
    # param_cov = Column("param_cov", postgresql.ARRAY(types.REAL))
    # fparam_cov = Column("fparam_cov", postgresql.ARRAY(types.REAL))
    teff = Column("teff", types.REAL)
    teff_err = Column("teff_err", types.REAL)
    logg = Column("logg", types.REAL)
    logg_err = Column("logg_err", types.REAL)
    vmicro = Column("vmicro", types.REAL)
    vmacro = Column("vmacro", types.REAL)
    vsini = Column("vsini", types.REAL)
    m_h = Column("m_h", types.REAL)
    m_h_err = Column("m_h_err", types.REAL)
    alpha_m = Column("alpha_m", types.REAL)
    alpha_m_err = Column("alpha_m_err", types.REAL)
    aspcap_chi2 = Column("aspcap_chi2", types.REAL)
    aspcap_class = Column("aspcap_class", types.String)
    aspcapflag = Column("aspcapflag", types.Integer)
    aspcapflags = Column("aspcapflags", types.String)
    # paramflag = Column("paramflag", postgresql.ARRAY(types.Integer))
    # felem = Column("felem", postgresql.ARRAY(types.REAL))
    # felem_err = Column("felem_err", postgresql.ARRAY(types.REAL))
    # x_h = Column("x_h", postgresql.ARRAY(types.REAL))
    # x_h_err = Column("x_h_err", postgresql.ARRAY(types.REAL))
    # x_m = Column("x_m", postgresql.ARRAY(types.REAL))
    # x_m_err = Column("x_m_err", postgresql.ARRAY(types.REAL))
    c_fe = Column("c_fe", types.REAL)
    ci_fe = Column("ci_fe", types.REAL)
    n_fe = Column("n_fe", types.REAL)
    o_fe = Column("o_fe", types.REAL)
    na_fe = Column("na_fe", types.REAL)
    mg_fe = Column("mg_fe", types.REAL)
    al_fe = Column("al_fe", types.REAL)
    si_fe = Column("si_fe", types.REAL)
    p_fe = Column("p_fe", types.REAL)
    s_fe = Column("s_fe", types.REAL)
    k_fe = Column("k_fe", types.REAL)
    ca_fe = Column("ca_fe", types.REAL)
    ti_fe = Column("ti_fe", types.REAL)
    tiii_fe = Column("tiii_fe", types.REAL)
    v_fe = Column("v_fe", types.REAL)
    cr_fe = Column("cr_fe", types.REAL)
    mn_fe = Column("mn_fe", types.REAL)
    fe_h = Column("fe_h", types.REAL)
    co_fe = Column("co_fe", types.REAL)
    ni_fe = Column("ni_fe", types.REAL)
    cu_fe = Column("cu_fe", types.REAL)
    ge_fe = Column("ge_fe", types.REAL)
    rb_fe = Column("rb_fe", types.REAL)
    y_fe = Column("y_fe", types.REAL)
    nd_fe = Column("nd_fe", types.REAL)
    c_fe_err = Column("c_fe_err", types.REAL)
    ci_fe_err = Column("ci_fe_err", types.REAL)
    n_fe_err = Column("n_fe_err", types.REAL)
    o_fe_err = Column("o_fe_err", types.REAL)
    na_fe_err = Column("na_fe_err", types.REAL)
    mg_fe_err = Column("mg_fe_err", types.REAL)
    al_fe_err = Column("al_fe_err", types.REAL)
    si_fe_err = Column("si_fe_err", types.REAL)
    p_fe_err = Column("p_fe_err", types.REAL)
    s_fe_err = Column("s_fe_err", types.REAL)
    k_fe_err = Column("k_fe_err", types.REAL)
    ca_fe_err = Column("ca_fe_err", types.REAL)
    ti_fe_err = Column("ti_fe_err", types.REAL)
    tiii_fe_err = Column("tiii_fe_err", types.REAL)
    v_fe_err = Column("v_fe_err", types.REAL)
    cr_fe_err = Column("cr_fe_err", types.REAL)
    mn_fe_err = Column("mn_fe_err", types.REAL)
    fe_h_err = Column("fe_h_err", types.REAL)
    co_fe_err = Column("co_fe_err", types.REAL)
    ni_fe_err = Column("ni_fe_err", types.REAL)
    cu_fe_err = Column("cu_fe_err", types.REAL)
    ge_fe_err = Column("ge_fe_err", types.REAL)
    rb_fe_err = Column("rb_fe_err", types.REAL)
    y_fe_err = Column("y_fe_err", types.REAL)
    nd_fe_err = Column("nd_fe_err", types.REAL)
    c_fe_flag = Column("c_fe_flag", types.Integer)
    ci_fe_flag = Column("ci_fe_flag", types.Integer)
    n_fe_flag = Column("n_fe_flag", types.Integer)
    o_fe_flag = Column("o_fe_flag", types.Integer)
    na_fe_flag = Column("na_fe_flag", types.Integer)
    mg_fe_flag = Column("mg_fe_flag", types.Integer)
    al_fe_flag = Column("al_fe_flag", types.Integer)
    si_fe_flag = Column("si_fe_flag", types.Integer)
    p_fe_flag = Column("p_fe_flag", types.Integer)
    s_fe_flag = Column("s_fe_flag", types.Integer)
    k_fe_flag = Column("k_fe_flag", types.Integer)
    ca_fe_flag = Column("ca_fe_flag", types.Integer)
    ti_fe_flag = Column("ti_fe_flag", types.Integer)
    tiii_fe_flag = Column("tiii_fe_flag", types.Integer)
    v_fe_flag = Column("v_fe_flag", types.Integer)
    cr_fe_flag = Column("cr_fe_flag", types.Integer)
    mn_fe_flag = Column("mn_fe_flag", types.Integer)
    fe_h_flag = Column("fe_h_flag", types.Integer)
    co_fe_flag = Column("co_fe_flag", types.Integer)
    ni_fe_flag = Column("ni_fe_flag", types.Integer)
    cu_fe_flag = Column("cu_fe_flag", types.Integer)
    ge_fe_flag = Column("ge_fe_flag", types.Integer)
    rb_fe_flag = Column("rb_fe_flag", types.Integer)
    y_fe_flag = Column("y_fe_flag", types.Integer)
    nd_fe_flag = Column("nd_fe_flag", types.Integer)
    # elem_chi2 = Column("elem_chi2", postgresql.ARRAY(types.REAL))
    # elemflag = Column("elemflag", postgresql.ARRAY(types.Integer))
    reduction_id = Column("reduction_id", types.String)
    src_h = Column("src_h", types.String)
    wash_m = Column("wash_m", types.REAL)
    wash_m_err = Column("wash_m_err", types.REAL)
    wash_t2 = Column("wash_t2", types.REAL)
    wash_t2_err = Column("wash_t2_err", types.REAL)
    ddo51 = Column("ddo51", types.REAL)
    ddo51_err = Column("ddo51_err", types.REAL)
    irac_3_6 = Column("irac_3_6", types.REAL)
    irac_3_6_err = Column("irac_3_6_err", types.REAL)
    irac_4_5 = Column("irac_4_5", types.REAL)
    irac_4_5_err = Column("irac_4_5_err", types.REAL)
    irac_5_8 = Column("irac_5_8", types.REAL)
    irac_5_8_err = Column("irac_5_8_err", types.REAL)
    irac_8_0 = Column("irac_8_0", types.REAL)
    irac_8_0_err = Column("irac_8_0_err", types.REAL)
    wise_4_5 = Column("wise_4_5", types.REAL)
    wise_4_5_err = Column("wise_4_5_err", types.REAL)
    targ_4_5 = Column("targ_4_5", types.REAL)
    targ_4_5_err = Column("targ_4_5_err", types.REAL)
    ak_targ = Column("ak_targ", types.REAL)
    ak_targ_method = Column("ak_targ_method", types.String)
    ak_wise = Column("ak_wise", types.REAL)
    sfd_ebv = Column("sfd_ebv", types.REAL)
    wash_ddo51_giant_flag = Column("wash_ddo51_giant_flag", types.SmallInteger)
    wash_ddo51_star_flag = Column("wash_ddo51_star_flag", types.SmallInteger)
    pmra = Column("pmra", types.REAL)
    pmdec = Column("pmdec", types.REAL)
    pm_src = Column("pm_src", types.String)
    # fparam_class = Column("fparam_class", postgresql.ARRAY(types.REAL))
    # chi2_class = Column("chi2_class", postgresql.ARRAY(types.REAL))

    def __repr__(self):
        return ("<ApogeeStar(id='{0}', apogee_id='{1}', {2} results)>"
                .format(self.id, self.apogee_id, len(self.results)))

    def apogeervdata(self):
        """Return a `twoface.data.APOGEERVData` instance for this star. """
        return star_to_apogeervdata(self)

class AllVisit(Base):
    __tablename__ = 'allvisit'

    id = Column(types.Integer, primary_key=True)

    # Columns from the FITS file, auto-generated by
    #       scripts/_generate_model_columns.py
    visit_id = Column("visit_id", types.String, index=True, unique=True)
    apred_version = Column("apred_version", types.String)
    apogee_id = Column("apogee_id", types.String, index=True)
    target_id = Column("target_id", types.String)
    file = Column("file", types.String)
    fiberid = Column("fiberid", types.SmallInteger)
    plate = Column("plate", types.String)
    mjd = Column("mjd", types.Integer)
    telescope = Column("telescope", types.String)
    location_id = Column("location_id", types.SmallInteger)
    ra = Column("ra", types.REAL)
    dec = Column("dec", types.REAL)
    glon = Column("glon", types.REAL)
    glat = Column("glat", types.REAL)
    j = Column("j", types.REAL)
    j_err = Column("j_err", types.REAL)
    h = Column("h", types.REAL)
    h_err = Column("h_err", types.REAL)
    k = Column("k", types.REAL)
    k_err = Column("k_err", types.REAL)
    ra_targ = Column("ra_targ", types.REAL)
    dec_targ = Column("dec_targ", types.REAL)
    apogee_target1 = Column("apogee_target1", types.Integer)
    apogee_target2 = Column("apogee_target2", types.Integer)
    apogee_target3 = Column("apogee_target3", types.Integer)
    targflags = Column("targflags", types.String)
    snr = Column("snr", types.REAL)
    starflag = Column("starflag", types.Integer)
    starflags = Column("starflags", types.String)
    dateobs = Column("dateobs", types.String)
    jd = Column("jd", types.REAL)
    bc = Column("bc", types.REAL)
    vtype = Column("vtype", types.SmallInteger)
    vrel = Column("vrel", types.REAL)
    vrelerr = Column("vrelerr", types.REAL)
    vhelio = Column("vhelio", types.REAL)
    vlsr = Column("vlsr", types.REAL)
    vgsr = Column("vgsr", types.REAL)
    chisq = Column("chisq", types.REAL)
    rv_teff = Column("rv_teff", types.REAL)
    rv_feh = Column("rv_feh", types.REAL)
    rv_logg = Column("rv_logg", types.REAL)
    rv_alpha = Column("rv_alpha", types.REAL)
    rv_carb = Column("rv_carb", types.REAL)
    synthfile = Column("synthfile", types.String)
    estvtype = Column("estvtype", types.SmallInteger)
    estvrel = Column("estvrel", types.REAL)
    estvrelerr = Column("estvrelerr", types.REAL)
    estvhelio = Column("estvhelio", types.REAL)
    synthvrel = Column("synthvrel", types.REAL)
    synthvrelerr = Column("synthvrelerr", types.REAL)
    synthvhelio = Column("synthvhelio", types.REAL)
    field = Column("field", types.String)
    commiss = Column("commiss", types.SmallInteger)
    extratarg = Column("extratarg", types.SmallInteger)
    min_h = Column("min_h", types.REAL)
    max_h = Column("max_h", types.REAL)
    min_jk = Column("min_jk", types.REAL)
    reduction_id = Column("reduction_id", types.String)
    src_h = Column("src_h", types.String)
    wash_m = Column("wash_m", types.REAL)
    wash_m_err = Column("wash_m_err", types.REAL)
    wash_t2 = Column("wash_t2", types.REAL)
    wash_t2_err = Column("wash_t2_err", types.REAL)
    ddo51 = Column("ddo51", types.REAL)
    ddo51_err = Column("ddo51_err", types.REAL)
    irac_3_6 = Column("irac_3_6", types.REAL)
    irac_3_6_err = Column("irac_3_6_err", types.REAL)
    irac_4_5 = Column("irac_4_5", types.REAL)
    irac_4_5_err = Column("irac_4_5_err", types.REAL)
    irac_5_8 = Column("irac_5_8", types.REAL)
    irac_5_8_err = Column("irac_5_8_err", types.REAL)
    irac_8_0 = Column("irac_8_0", types.REAL)
    irac_8_0_err = Column("irac_8_0_err", types.REAL)
    wise_4_5 = Column("wise_4_5", types.REAL)
    wise_4_5_err = Column("wise_4_5_err", types.REAL)
    targ_4_5 = Column("targ_4_5", types.REAL)
    targ_4_5_err = Column("targ_4_5_err", types.REAL)
    ak_targ = Column("ak_targ", types.REAL)
    ak_targ_method = Column("ak_targ_method", types.String)
    ak_wise = Column("ak_wise", types.REAL)
    sfd_ebv = Column("sfd_ebv", types.REAL)
    wash_ddo51_giant_flag = Column("wash_ddo51_giant_flag", types.SmallInteger)
    wash_ddo51_star_flag = Column("wash_ddo51_star_flag", types.SmallInteger)
    pmra = Column("pmra", types.REAL)
    pmdec = Column("pmdec", types.REAL)
    pm_src = Column("pm_src", types.String)

    def __repr__(self):
        return "<ApogeeVisit(APOGEE_ID='{}', MJD='{}')>".format(self.apogee_id, self.mjd)

class RedClump(Base):
    __tablename__ = 'redclump'

    id = Column(types.Integer, primary_key=True)

    allstar_id = Column("allstar_id", types.Integer,
                        ForeignKey('allstar.id', ondelete='CASCADE'),
                        index=True)
    star = relationship("AllStar",
                        backref=backref("red_clump",
                                        cascade="all, delete-orphan"))

    # These are in AllStar
    # apstar_id = Column("apstar_id", types.String)
    # aspcap_id = Column("aspcap_id", types.String)
    # apogee_id = Column("apogee_id", types.String)
    # telescope = Column("telescope", types.String)
    # location_id = Column("location_id", types.SmallInteger)
    # field = Column("field", types.String)
    # j = Column("j", types.REAL)
    # j_err = Column("j_err", types.REAL)
    # h = Column("h", types.REAL)
    # h_err = Column("h_err", types.REAL)
    # k = Column("k", types.REAL)
    # k_err = Column("k_err", types.REAL)
    # ra = Column("ra", types.Numeric)
    # dec = Column("dec", types.Numeric)
    # glon = Column("glon", types.Numeric)
    # glat = Column("glat", types.Numeric)
    # apogee_target1 = Column("apogee_target1", types.Integer)
    # apogee_target2 = Column("apogee_target2", types.Integer)
    # apogee_target3 = Column("apogee_target3", types.Integer)
    # targflags = Column("targflags", types.String)
    # nvisits = Column("nvisits", types.Integer)
    # commiss = Column("commiss", types.SmallInteger)
    # snr = Column("snr", types.REAL)
    # starflag = Column("starflag", types.Integer)
    # starflags = Column("starflags", types.String)
    # andflag = Column("andflag", types.Integer)
    # andflags = Column("andflags", types.String)
    # vhelio_avg = Column("vhelio_avg", types.REAL)
    # vscatter = Column("vscatter", types.REAL)
    # verr = Column("verr", types.REAL)
    # verr_med = Column("verr_med", types.REAL)
    # meanfib = Column("meanfib", types.REAL)
    # sigfib = Column("sigfib", types.REAL)
    # snrev = Column("snrev", types.REAL)
    # extratarg = Column("extratarg", types.SmallInteger)
    # min_h = Column("min_h", types.REAL)
    # max_h = Column("max_h", types.REAL)
    # min_jk = Column("min_jk", types.REAL)
    # param = Column("param", postgresql.ARRAY(types.REAL))
    # fparam = Column("fparam", postgresql.ARRAY(types.REAL))
    # param_cov = Column("param_cov", postgresql.ARRAY(types.REAL))
    # fparam_cov = Column("fparam_cov", postgresql.ARRAY(types.REAL))
    # teff = Column("teff", types.REAL)
    # teff_err = Column("teff_err", types.REAL)
    # logg = Column("logg", types.REAL)
    # logg_err = Column("logg_err", types.REAL)
    # vmicro = Column("vmicro", types.REAL)
    # vmacro = Column("vmacro", types.REAL)
    # vsini = Column("vsini", types.REAL)
    # m_h = Column("m_h", types.REAL)
    # m_h_err = Column("m_h_err", types.REAL)
    # alpha_m = Column("alpha_m", types.REAL)
    # alpha_m_err = Column("alpha_m_err", types.REAL)
    # aspcap_chi2 = Column("aspcap_chi2", types.REAL)
    # aspcap_class = Column("aspcap_class", types.String)
    # aspcapflag = Column("aspcapflag", types.Integer)
    # aspcapflags = Column("aspcapflags", types.String)
    # paramflag = Column("paramflag", postgresql.ARRAY(types.Integer))
    # felem = Column("felem", postgresql.ARRAY(types.REAL))
    # felem_err = Column("felem_err", postgresql.ARRAY(types.REAL))
    # x_h = Column("x_h", postgresql.ARRAY(types.REAL))
    # x_h_err = Column("x_h_err", postgresql.ARRAY(types.REAL))
    # x_m = Column("x_m", postgresql.ARRAY(types.REAL))
    # x_m_err = Column("x_m_err", postgresql.ARRAY(types.REAL))
    # c_fe = Column("c_fe", types.REAL)
    # ci_fe = Column("ci_fe", types.REAL)
    # n_fe = Column("n_fe", types.REAL)
    # o_fe = Column("o_fe", types.REAL)
    # na_fe = Column("na_fe", types.REAL)
    # mg_fe = Column("mg_fe", types.REAL)
    # al_fe = Column("al_fe", types.REAL)
    # si_fe = Column("si_fe", types.REAL)
    # p_fe = Column("p_fe", types.REAL)
    # s_fe = Column("s_fe", types.REAL)
    # k_fe = Column("k_fe", types.REAL)
    # ca_fe = Column("ca_fe", types.REAL)
    # ti_fe = Column("ti_fe", types.REAL)
    # tiii_fe = Column("tiii_fe", types.REAL)
    # v_fe = Column("v_fe", types.REAL)
    # cr_fe = Column("cr_fe", types.REAL)
    # mn_fe = Column("mn_fe", types.REAL)
    # fe_h = Column("fe_h", types.REAL)
    # co_fe = Column("co_fe", types.REAL)
    # ni_fe = Column("ni_fe", types.REAL)
    # cu_fe = Column("cu_fe", types.REAL)
    # ge_fe = Column("ge_fe", types.REAL)
    # rb_fe = Column("rb_fe", types.REAL)
    # y_fe = Column("y_fe", types.REAL)
    # nd_fe = Column("nd_fe", types.REAL)
    # c_fe_err = Column("c_fe_err", types.REAL)
    # ci_fe_err = Column("ci_fe_err", types.REAL)
    # n_fe_err = Column("n_fe_err", types.REAL)
    # o_fe_err = Column("o_fe_err", types.REAL)
    # na_fe_err = Column("na_fe_err", types.REAL)
    # mg_fe_err = Column("mg_fe_err", types.REAL)
    # al_fe_err = Column("al_fe_err", types.REAL)
    # si_fe_err = Column("si_fe_err", types.REAL)
    # p_fe_err = Column("p_fe_err", types.REAL)
    # s_fe_err = Column("s_fe_err", types.REAL)
    # k_fe_err = Column("k_fe_err", types.REAL)
    # ca_fe_err = Column("ca_fe_err", types.REAL)
    # ti_fe_err = Column("ti_fe_err", types.REAL)
    # tiii_fe_err = Column("tiii_fe_err", types.REAL)
    # v_fe_err = Column("v_fe_err", types.REAL)
    # cr_fe_err = Column("cr_fe_err", types.REAL)
    # mn_fe_err = Column("mn_fe_err", types.REAL)
    # fe_h_err = Column("fe_h_err", types.REAL)
    # co_fe_err = Column("co_fe_err", types.REAL)
    # ni_fe_err = Column("ni_fe_err", types.REAL)
    # cu_fe_err = Column("cu_fe_err", types.REAL)
    # ge_fe_err = Column("ge_fe_err", types.REAL)
    # rb_fe_err = Column("rb_fe_err", types.REAL)
    # y_fe_err = Column("y_fe_err", types.REAL)
    # nd_fe_err = Column("nd_fe_err", types.REAL)
    # c_fe_flag = Column("c_fe_flag", types.Integer)
    # ci_fe_flag = Column("ci_fe_flag", types.Integer)
    # n_fe_flag = Column("n_fe_flag", types.Integer)
    # o_fe_flag = Column("o_fe_flag", types.Integer)
    # na_fe_flag = Column("na_fe_flag", types.Integer)
    # mg_fe_flag = Column("mg_fe_flag", types.Integer)
    # al_fe_flag = Column("al_fe_flag", types.Integer)
    # si_fe_flag = Column("si_fe_flag", types.Integer)
    # p_fe_flag = Column("p_fe_flag", types.Integer)
    # s_fe_flag = Column("s_fe_flag", types.Integer)
    # k_fe_flag = Column("k_fe_flag", types.Integer)
    # ca_fe_flag = Column("ca_fe_flag", types.Integer)
    # ti_fe_flag = Column("ti_fe_flag", types.Integer)
    # tiii_fe_flag = Column("tiii_fe_flag", types.Integer)
    # v_fe_flag = Column("v_fe_flag", types.Integer)
    # cr_fe_flag = Column("cr_fe_flag", types.Integer)
    # mn_fe_flag = Column("mn_fe_flag", types.Integer)
    # fe_h_flag = Column("fe_h_flag", types.Integer)
    # co_fe_flag = Column("co_fe_flag", types.Integer)
    # ni_fe_flag = Column("ni_fe_flag", types.Integer)
    # cu_fe_flag = Column("cu_fe_flag", types.Integer)
    # ge_fe_flag = Column("ge_fe_flag", types.Integer)
    # rb_fe_flag = Column("rb_fe_flag", types.Integer)
    # y_fe_flag = Column("y_fe_flag", types.Integer)
    # nd_fe_flag = Column("nd_fe_flag", types.Integer)
    # elem_chi2 = Column("elem_chi2", postgresql.ARRAY(types.REAL))
    # elemflag = Column("elemflag", postgresql.ARRAY(types.Integer))
    # ak_targ = Column("ak_targ", types.REAL)
    # ak_targ_method = Column("ak_targ_method", types.String)
    # fparam_class = Column("fparam_class", postgresql.ARRAY(types.REAL))
    # chi2_class = Column("chi2_class", postgresql.ARRAY(types.REAL))
    j0 = Column("j0", types.Numeric)
    h0 = Column("h0", types.Numeric)
    k0 = Column("k0", types.Numeric)
    metals = Column("metals", types.Numeric)
    alphafe = Column("alphafe", types.Numeric)
    addl_logg_cut = Column("addl_logg_cut", types.Integer)
    rc_dist = Column("rc_dist", types.Numeric)
    rc_dm = Column("rc_dm", types.Numeric)
    rc_galr = Column("rc_galr", types.Numeric)
    rc_galphi = Column("rc_galphi", types.Numeric)
    rc_galz = Column("rc_galz", types.Numeric)
    tyc2match = Column("tyc2match", types.Integer)
    tyc1 = Column("tyc1", types.Integer)
    tyc2 = Column("tyc2", types.Integer)
    tyc3 = Column("tyc3", types.Integer)
    stat = Column("stat", types.Integer)
    invsf = Column("invsf", types.Numeric)
    pmra = Column("pmra", types.Numeric)
    pmdec = Column("pmdec", types.Numeric)
    pmra_err = Column("pmra_err", types.Numeric)
    pmdec_err = Column("pmdec_err", types.Numeric)
    pmmatch = Column("pmmatch", types.Integer)
    galvr = Column("galvr", types.Numeric)
    galvt = Column("galvt", types.Numeric)
    galvz = Column("galvz", types.Numeric)
    pmra_ppmxl = Column("pmra_ppmxl", types.Numeric)
    pmdec_ppmxl = Column("pmdec_ppmxl", types.Numeric)
    pmra_err_ppmxl = Column("pmra_err_ppmxl", types.Numeric)
    pmdec_err_ppmxl = Column("pmdec_err_ppmxl", types.Numeric)
    pmmatch_ppmxl = Column("pmmatch_ppmxl", types.Integer)
    galvr_ppmxl = Column("galvr_ppmxl", types.Numeric)
    galvt_ppmxl = Column("galvt_ppmxl", types.Numeric)
    galvz_ppmxl = Column("galvz_ppmxl", types.Numeric)

    def __repr__(self):
        return ("<RedClump(id='{0}', apogee_id='{1}', {2} results)>"
                .format(self.id, self.star.apogee_id, len(self.star.results)))

    def apogeervdata(self):
        """Return a `twoface.data.APOGEERVData` instance for this star."""
        return star_to_apogeervdata(self.star)

JitterType = QuantityTypeClassFactory(u.m/u.s)
PeriodType = QuantityTypeClassFactory(u.day)
class JokerRun(Base):
    __tablename__ = 'jokerrun'

    id = Column(types.Integer, primary_key=True)
    date = Column('date', types.DateTime, default=datetime.datetime.utcnow)
    name = Column('name', types.String, nullable=False)
    notes = Column('notes', types.String)
    config_file = Column('config_file', types.String, nullable=False)
    # cache_path = Column('cache_path', types.String, nullable=False) # named after joker run name
    prior_samples_file = Column('prior_samples_file', types.String,
                                nullable=False)

    stars = relationship("AllStar", secondary="starresult",
                         back_populates="joker_runs")

    # The Joker parameters:
    jitter = Column('jitter', JitterType, default=float('nan')) # nan=not fixed
    jitter_mean = Column('jitter_mean', types.Numeric, default=float('nan'))
    jitter_stddev = Column('jitter_stddev', types.Numeric, default=float('nan'))
    jitter_unit = Column('jitter_unit', types.String, default="")

    P_min = Column('P_min', PeriodType, nullable=False)
    P_max = Column('P_max', PeriodType, nullable=False)
    requested_samples_per_star = Column('requested_samples_per_star',
                                        types.Integer, nullable=False)
    max_prior_samples = Column('max_prior_samples', types.Integer,
                               nullable=False)

    def __repr__(self):
        return ("<JokerRun(name='{0}', date='{0}')>"
                .format(self.name, self.date.isoformat()))

class NessRG(Base):
    __tablename__ = 'nessrg'

    id = Column(types.Integer, primary_key=True)

    allstar_id = Column('allstar_id', types.Integer,
                        ForeignKey('allstar.id', ondelete='CASCADE'),
                        index=True)
    star = relationship("AllStar", backref=backref("ness_rg", uselist=False))

    # Melissa Ness' masses and ages from:
    # http://iopscience.iop.org/article/10.3847/0004-637X/823/2/114/meta

    # strip ()[], replace / with _
    Type = Column('Type', types.Integer)
    Teff = Column('Teff', types.REAL)
    logg = Column('logg', types.REAL)
    Fe_H = Column('Fe_H', types.REAL)
    a_Fe = Column('a_Fe', types.REAL)
    lnM = Column('lnM', types.REAL)
    lnAge = Column('lnAge', types.REAL)
    chi2 = Column('chi2', types.REAL)

    e_Teff = Column('e_Teff', types.REAL)
    e_logg = Column('e_logg', types.REAL)
    e_Fe_H = Column('e_Fe_H', types.REAL)
    e_a_Fe = Column('e_a_Fe', types.REAL)
    e_logM = Column('e_logM', types.REAL)
    e_logAge = Column('e_logAge', types.REAL)
