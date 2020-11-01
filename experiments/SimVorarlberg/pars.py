import sciris as sc

pars = pars = sc.objdict(
    pop_size        = 40000,
    pop_infected    = 10,
    pop_type        = 'synthpops',
    location        = 'Vorarlberg',
    n_days          = 180,
    verbose         = 1,
    pop_scale       = 10,
    n_beds_hosp     = 700 ,  #source: http://www.kaz.bmg.gv.at/fileadmin/user_upload/Betten/1_T_Betten_SBETT.pdf (2019)
    n_beds_icu      = 30,      # source: https://vbgv1.orf.at/stories/493214 (2011 no recent data found)
    iso_factor      = dict(h=1, s=1, w=1, c=1)
)