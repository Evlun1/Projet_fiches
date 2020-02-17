import pulp
import numpy as np
import pandas as pd
import itertools
import time
import os


def optimization():
    # Chemin d'accès des fichiers
    path_excel = 'Input/AS/'
    path_ae = 'Input/AE/'
    name_excel_as = '201909_Inputs_MD_v20190902_v2.xlsx'
    input_booster = "Booster.xlsx"
    input_ae = '201909_Inputs_SC_Satisfaction_RF2.xlsx'
    input_vol_ae = "201910_Inputs_SC_Volumes_RF2.xlsx"
    output = 'Output/'

    template = 'Input/AS/excel_template_allocation_flux V3.xlsx'

    # Liste des différentes contraintes pour la création des scénarios (à modifier)

    # lecture de la feuille contraintes contenant le cadre de l'optimisation
    template_contraintes = pd.read_excel(template, sheet_name="Mois & autres")
    template_contraintes.head()

    # liste des contraintes et leur valeur
    list_contraintes = ['list_bool_isobudget',
                        'list_seuil_VN_prospect_min',
                        'list_seuil_md',
                        'list_seuil_ret',
                        'list_seuil_SC',
                        'list_seuil_FV_inshore',
                        'list_seuil_satis',
                        'list_bool_capacitaire']

    list_scenario = []
    for i in list_contraintes:
        list_scenario.append(float(template_contraintes[template_contraintes['nom_contraintes'] == i]
                                   ['valeur_contraintes']))

    # Prise en compte des AS/AE

    AE = float(template_contraintes[template_contraintes['nom_contraintes'] == 'AE']
               ['valeur_contraintes'])  # 1 si AE pris en compte, 0 sinon
    AS = float(template_contraintes[template_contraintes['nom_contraintes'] == 'AS']
               ['valeur_contraintes'])  # 1 si AS pris en compte, 0 sinon

    if AE == 0:
        print('AE non pris en compte')
    if AS == 0:
        print('AS non pris en compte')

    # liste des mois ou le programme doit tourner
    list_mois = list(template_contraintes['mois'] + '_'
                     + template_contraintes['année'].dropna().astype(int).astype(str))
    list_mois = [x for x in list_mois if str(x) != 'nan']

    # budget prévisionnel
    budget_prev_2019 = int(template_contraintes[template_contraintes['nom_contraintes'] == 'budget_prev_2019']
                           ['valeur_contraintes'])

    # ratio de budget par mois
    ratio_mois_budget = template_contraintes[template_contraintes['nom_contraintes']
                                             == 'ratio_mois_budget']['valeur_contraintes']
    ratio_mois_budget = list(ratio_mois_budget)[0].split(',')
    ratio_mois_budget = [int(x) for x in ratio_mois_budget]

    if len(ratio_mois_budget) != len(list_mois):
        ratio_mois_budget = np.repeat(100 / len(list_mois), len(list_mois))

    list_budget_prev_2019 = []
    for ratio_mois in ratio_mois_budget:
        list_budget_prev_2019.append(budget_prev_2019 * ratio_mois / 100)
    print(sum(list_budget_prev_2019))

    # VN prev--> ??
    VN_prev = template_contraintes[template_contraintes['nom_contraintes']
                                   == 'VN_prev']['valeur_contraintes']
    VN_prev = list(VN_prev)[0].split(',')
    VN_prev = [int(x) for x in VN_prev]
    VN = 1

    if len(VN_prev) != len(list_mois):
        print('contraintes sur VN abandonnées')
        VN = 0

    # Programme

    # on récupère le capacitaire des centre
    template_centre_capacitaire = pd.read_excel(template, sheet_name=3)
    template_centre_capacitaire.head()

    # on récupère les noms de colonnes
    template_nom_colonne = pd.read_excel(template, sheet_name="nom_colonne")
    nom_colonne_AS = list(template_nom_colonne['AS'])
    nom_colonne_AE = list(template_nom_colonne['AE'])

    # on récupère l'onglet principal as avec les infos de perf
    template_perf_as = pd.read_excel(template, sheet_name=0)
    flux_as = list(template_perf_as.iloc[:, 0])
    template_perf_as.head()

    # on récupère l'onglet principal ae avec les infos de perf
    template_perf_ae = pd.read_excel(template, sheet_name=1)
    competence = list(template_perf_ae.iloc[:, 0])
    template_perf_ae.head()

    # colonne avec VN_FL
    # mettre des arrêts?
    if AS == 1:  # on ne le fait que si les AS sont pris en compte
        perf = template_perf_as[[x for x in list(template_perf_as.columns)
                                 if nom_colonne_AS[0] in x]]
        perf = perf.copy()
        perf['Flux'] = flux_as
        # list(map(lambda x: str(x).replace(nom_colonne_AS[0]+'_',''),list(template_VN_FL.columns)))
        perf.columns = [str(x).replace(nom_colonne_AS[0] + '_', '')
                        for x in list(perf.columns)]
        perf = perf.melt(id_vars='Flux', value_name='Perf')
        perf.columns = ['flux', 'centres', 'perf']

        nb_centre_perf = len(perf['centres'].unique())
        if perf.shape[0] % template_centre_capacitaire[
            template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] != 0:
            print(f'Pb sur le nombre de centres / manque un centre perf: '
                  f'{template_centre_capacitaire[template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] - nb_centre_perf}')
        else:
            print(f'taille de perf as: {perf.shape}')
    else:
        print('AS non pris en compte')

    # colonne ae avec res_satis
    # mettre des arrêts?

    if AE == 1:

        perf_ae = template_perf_ae[[x for x in list(template_perf_ae.columns)
                                    if nom_colonne_AE[1] in x]]
        perf_ae = perf_ae.copy()
        perf_ae['flux'] = competence
        # list(map(lambda x: str(x).replace(nom_colonne_AS[0]+'_',''),list(template_VN_FL.columns)))
        perf_ae.columns = [str(x).replace(nom_colonne_AE[1] + '_', '')
                           for x in list(perf_ae.columns)]
        perf_ae = perf_ae.melt(id_vars='flux', value_name='Perf')
        perf_ae.columns = ['flux', 'centres', 'perf']

        nb_centre_perf_ae = len(perf_ae['centres'].unique())
        if perf_ae.shape[0] % template_centre_capacitaire[
            template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] != 0:
            print(
                f'Pb sur le nombre de centres / manque un centre perf_ae: {template_centre_capacitaire[template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] - nb_centre_perf_ae}')
        else:
            print(f'taille de perf ae: {perf_ae.shape}')

    else:
        print('AE non pris en compte')

    if AE == 1 and AS == 1:
        perf = pd.concat([perf, perf_ae], sort=False)
        print(f'taille de perf: {perf.shape}')
    elif AE == 1 and AS == 0:
        perf = perf_ae
        print('AS non pris en compte')
    else:
        perf = perf
        print('AE non pris en compte')

    centres = perf.centres.unique()
    centres = sorted(centres)
    flux = perf.flux.unique()
    flux = sorted(flux)

    # colonne CA_FL
    # mettre des arrêts?

    if AS == 1:
        ca_fl = template_perf_as[[x for x in list(template_perf_as.columns) if nom_colonne_AS[1] in x]]
        ca_fl = ca_fl.copy()
        ca_fl['Flux'] = flux_as
        # list(map(lambda x: str(x).replace(nom_colonne_AS[0]+'_',''),list(template_VN_FL.columns)))
        ca_fl.columns = [str(x).replace(nom_colonne_AS[1] + '_', '') for x in list(ca_fl.columns)]

        ca_fl = ca_fl.melt(id_vars='Flux', value_name='CA_FL')
        ca_fl.columns = ['flux', 'centres', 'CA_FL']

        nb_centre_ca_fl = len(ca_fl['centres'].unique())
        if ca_fl.shape[0] % template_centre_capacitaire[
            template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] != 0:
            print(
                f'Pb sur le nombre de centres / manque un centre ca_fl: {template_centre_capacitaire[template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] - nb_centre_ca_fl}')
        else:
            print(f'taille de ca_fl as: {ca_fl.shape}')

    else:
        print('AS non pris en compte')

    # colonne pu
    # mettre des arrêts?

    if AS == 1:
        pu = template_perf_as[[x for x in list(template_perf_as.columns) if nom_colonne_AS[2] in x]]
        pu = pu.copy()
        pu['Flux'] = flux_as
        # list(map(lambda x: str(x).replace(nom_colonne_AS[0]+'_',''),list(template_VN_FL.columns)))
        pu.columns = [str(x).replace(nom_colonne_AS[2] + '_', '') for x in list(pu.columns)]

        pu = pu.melt(id_vars='Flux', value_name='pu')
        pu.columns = ['flux', 'centres', 'pu']

        nb_centre_pu = len(pu['centres'].unique())
        if pu.shape[0] % template_centre_capacitaire[
            template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] != 0:
            print(
                f'Pb sur le nombre de centres / manque un centre pu: {template_centre_capacitaire[template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] - nb_centre_pu}')
        else:
            print(f'taille de pu as: {pu.shape}')
    else:
        print('AS non pris en compte')

    # colonne ae tarif
    # mettre des arrêts?

    if AE == 1:
        pu_ae = template_perf_ae[[x for x in list(template_perf_ae.columns) if nom_colonne_AE[0] in x]]
        pu_ae = pu_ae.copy()
        pu_ae['flux'] = competence
        # list(map(lambda x: str(x).replace(nom_colonne_AS[0]+'_',''),list(template_VN_FL.columns)))
        pu_ae.columns = [str(x).replace(nom_colonne_AE[0] + '_', '') for x in list(pu_ae.columns)]
        pu_ae = pu_ae.melt(id_vars='flux', value_name='pu')
        pu_ae.columns = ['flux', 'centres', 'pu']

        nb_centre_pu_ae = len(pu_ae['centres'].unique())
        if pu_ae.shape[0] % template_centre_capacitaire[
            template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] != 0:
            print(
                f'Pb sur le nombre de centres / manque un centre pu_ae: {template_centre_capacitaire[template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] - nb_centre_pu_ae}')
        else:
            print(f'taille de pu_ae: {pu_ae.shape}')

        flux_ae = list(perf_ae.flux.unique())

    else:
        print('AE non pris en compte')

    if AE == 1 and AS == 1:
        pu = pd.concat([pu, pu_ae], sort=False)
        print(f'taille de pu: {pu.shape}')
    elif AE == 1 and AS == 0:
        pu = pu_ae
        print(f'taille de pu: {pu.shape}')
        print('AS non pris en compte')

    else:
        pu = pu
        print(f'taille de pu: {pu.shape}')
        print('AE non pris en compte')

    # Import de la valeur des flux AS
    valeur = template_perf_as[['flux', 'tx8.8%']]
    valeur.columns = ['flux', 'valeur']

    # colonne affectation
    # mettre des arrêts?

    if AS == 1:
        affectation = template_perf_as[[x for x in list(template_perf_as.columns) if nom_colonne_AS[3] in x]]
        affectation = affectation.copy()
        affectation['Flux'] = flux_as
        # list(map(lambda x: str(x).replace(nom_colonne_AS[0]+'_',''),list(template_VN_FL.columns)))
        affectation.columns = [str(x).replace(nom_colonne_AS[3] + '_', '') for x in list(affectation.columns)]

        affectation = affectation.melt(id_vars='Flux', value_name='affectation')
        affectation.columns = ['flux', 'centres', 'affectation']

        nb_centre_affectation = len(affectation['centres'].unique())
        if affectation.shape[0] % template_centre_capacitaire[
            template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] != 0:
            print(
                f'Pb sur le nombre de centres / manque un centre affectation: {template_centre_capacitaire[template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] - nb_centre_affectation}')
        else:
            print(f'taille de affectation as: {affectation.shape}')
    else:
        print('AS non pris en compte')

    if AE == 1:
        affectation_AE = perf_ae.fillna(0)
        affectation_AE.loc[affectation_AE['perf'] > 0, 'affectation'] = 1
        affectation_AE.loc[affectation_AE['perf'] == 0, 'affectation'] = 0
        affectation_AE = affectation_AE[['flux', 'centres', 'affectation']]
    else:
        print('AE non pris en compte')

    if AE == 1 and AS == 1:
        affectation = pd.concat([affectation, affectation_AE])
        print(f'taille de affectation: {affectation.shape}')

    elif AS == 1 and AE == 0:
        affectation = affectation
        print('AE non pris en compte')

    else:
        affectation = affectation_AE
        print('AS non pris en compte')

    # colonne seuil
    # mettre des arrêts?

    if AS == 1:
        seuil = template_perf_as[[x for x in list(template_perf_as.columns) if nom_colonne_AS[4] in x]]
        seuil = seuil.copy()
        seuil['Flux'] = flux_as
        # list(map(lambda x: str(x).replace(nom_colonne_AS[0]+'_',''),list(template_VN_FL.columns)))
        seuil.columns = [str(x).replace(nom_colonne_AS[4] + '_', '') for x in list(seuil.columns)]

        seuil = seuil.melt(id_vars='Flux', value_name='seuil')
        seuil.columns = ['flux', 'centres', 'seuil']

    nb_centre_seuil = len(seuil['centres'].unique())
    if seuil.shape[0] % template_centre_capacitaire[
        template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] != 0:
        print(
            f'Pb sur le nombre de centres / manque un centre affectation: {template_centre_capacitaire[template_centre_capacitaire[template_centre_capacitaire.columns[0]].notnull()].shape[0] - nb_centre_seuil}')
    else:
        print(f'taille de affectation as: {seuil.shape}')

    seuil = pd.merge(perf, seuil, how='inner', on=['flux', 'centres'])
    seuil['VN/FL/seuil'] = seuil.perf / seuil.seuil
    print(seuil.shape)

    seuil['VN/FL/seuil'] = round(seuil['VN/FL/seuil'], 2)
    # seuil['VN/FL/seuil'] = round(seuil['VN/FL/seuil'].astype(float),2)

    # check les .loc

    seuil.loc[seuil['VN/FL/seuil'] >= 2, 'VN/FL/seuil'] = 2
    # Utilisation du booster pour avoir un PU dynamique en fonction de la performance VN/FL
    booster_MD = pd.read_excel(path_excel + input_booster, sheet_name="Booster_MD")
    var_pu = pd.merge(seuil, booster_MD, how='left', on='VN/FL/seuil')
    print(var_pu.shape)
    booster_ret = pd.read_excel(path_excel + input_booster, sheet_name="Booster_Ret")
    var_pu = pd.merge(var_pu, booster_ret, how='left', on='VN/FL/seuil')
    print(var_pu.shape)
    var_pu['booster'] = 0
    var_pu.loc[(var_pu['flux'].str.contains('MD') & (var_pu['centres'].str.contains('CCA 1'))), 'booster'] = var_pu[
        'Surrémunération moyenne CCA']
    var_pu.loc[(var_pu['flux'].str.contains('MD') & (-var_pu['centres'].str.contains('CCA 1'))), 'booster'] = var_pu[
        'Surrémunération moyenne MD']
    var_pu.loc[var_pu['flux'].str.contains('ASR'), 'booster'] = var_pu['Surrémunération moyenne RET']
    var_pu = var_pu[-var_pu.perf.isna()]
    var_pu2 = var_pu[['flux', 'centres', 'booster']]

    # Jointure de tous les élements par flux * destination

    # si pas de pu en ae affectation quand meme??

    df2 = pd.DataFrame(list(itertools.product(centres, flux)))
    df2.columns = ['centres', 'flux']

    df2 = pd.merge(df2, perf, how='left', on=['centres', 'flux'])
    df2 = pd.merge(df2, ca_fl, how='left', on=['centres', 'flux'])
    df2 = pd.merge(df2, pu, how='left', on=['centres', 'flux'])
    df2 = pd.merge(df2, valeur, how='left', on='flux')
    df2 = pd.merge(df2, affectation, how='left', on=['centres', 'flux'])

    if AE == 1:
        df2.loc[df2['flux'].isin(flux_ae), 'CA_FL'] = 1
        df2.loc[df2['flux'].isin(flux_ae), 'perf'] = 1

        perf_ae.columns = ['centres', 'flux', 'Satisfaction']
        df2 = pd.merge(df2, perf_ae, how='left', on=['centres', 'flux'])
    else:
        print('AE non pris en compte')

    df2 = pd.merge(df2, var_pu2, how='left', on=['centres', 'flux'])

    df2.fillna(0, inplace=True)

    if AE == 1 and AS == 1:
        df2 = df2[['centres', 'flux', 'perf', 'valeur', 'CA_FL', 'affectation', 'pu', 'Satisfaction', 'booster']]
    elif AE == 0 and AS == 1:
        df2 = df2[['centres', 'flux', 'perf', 'valeur', 'CA_FL', 'affectation', 'pu', 'booster']]
        print('AE non pris en compte')
    else:
        print('AS non pris en compte - a check')

    df2['pu_booste'] = df2.pu + (df2.pu * df2.booster)
    df2['diff_pu'] = abs(df2.pu - df2.pu_booste)

    # Capacitaire par mois des différentes destinations
    M_centres = list(template_centre_capacitaire.Total_mensuel)
    print(f'Nombre de sites : {len(M_centres)}')
    print(f'capacitaire total : {sum(M_centres)}')

    # d'ou vient la liste FAI|BASE|DESABONNES|PROSPECTS|RECENTS?
    if AS == 1:
        list_SC_mono_site = template_contraintes[template_contraintes['list_SC_mono_site'].notnull()][
            'list_SC_mono_site']
        list_SC_mono_site = '|'.join(list(list_SC_mono_site))

        list_prospect_index = []
        df2_prospect = df2[df2.flux.str.contains(list_SC_mono_site, regex=True)]
        list_prospect_index = list(df2_prospect.index.values)

    if AE == 1:
        list_ae_index = []
        df2_ae = df2[df2.flux.isin(flux_ae)]
        list_ae_index = list(df2_ae.index.values)

    FV_inshore = template_contraintes[template_contraintes['list_FV_inshore'].notnull()]['list_FV_inshore']
    FV_inshore = '|'.join(list(FV_inshore))

    list_FV_inshore = []
    df2_FV_inshore = df2[(df2.flux.str.contains(FV_inshore, regex=True)) & (df2.centres.str.contains('1', regex=True))]
    list_FV_inshore = list(df2_FV_inshore.index.values)

    # Nombre de fiches à délivrer par mois

    if AS == 1:
        N_fiches_input = template_perf_as[[x for x in template_perf_as.columns if x in list_mois]]

        mois_nul = pd.DataFrame(
            dict(name=list(N_fiches_input.isnull().sum().index), valeur=list(N_fiches_input.isnull().sum())))
        N_fiches_input = N_fiches_input.drop(list(mois_nul[mois_nul['valeur'] == N_fiches_input.shape[0]]['name']),
                                             axis=1)

        N_fiches_input = N_fiches_input.copy()
        N_fiches_input['flux'] = flux_as
        N_fiches_input = N_fiches_input.sort_values(['flux'])

    if AE == 1:
        fiches_ae = template_perf_ae[[x for x in template_perf_as.columns if x in list_mois]]

        mois_nul = pd.DataFrame(dict(name=list(fiches_ae.isnull().sum().index), valeur=list(fiches_ae.isnull().sum())))
        fiches_ae = fiches_ae.drop(list(mois_nul[mois_nul['valeur'] == fiches_ae.shape[0]]['name']), axis=1)

        fiches_ae = fiches_ae.copy()
        fiches_ae['flux'] = competence

    if AE == 1 and AS == 1:
        N_fiches_input_2 = pd.concat([N_fiches_input, fiches_ae])
        N_fiches_input = N_fiches_input_2.sort_values(['flux'])
    elif AE == 0 and AS == 1:
        N_fiches_input = N_fiches_input.sort_values(['flux'])
        print('AE non pris en compte')
    else:
        N_fiches_input_2 = fiches_ae
        N_fiches_input = N_fiches_input_2.sort_values(['flux'])
        print('AS non pris en compte')

    if N_fiches_input.iloc[0, 0] == str:
        for i in N_fiches_input.drop('flux', axis=1).columns:
            N_fiches_input[i] = N_fiches_input[i].str.replace(' ', '')
            N_fiches_input[i] = N_fiches_input[i].fillna(0)
            N_fiches_input[i] = N_fiches_input[i].astype(int)

    N_fiches_FV = N_fiches_input[N_fiches_input.flux.str.contains(FV_inshore, regex=True)]

    list_SC_mono_site = list(
        template_contraintes[template_contraintes['list_SC_mono_site'].notnull()]['list_SC_mono_site'])

    res_detaille_final = []

    list_scenario = [list_scenario]

    combinaison = 1
    scenarios_non_optimal = []
    for scenario in list_scenario:
        bool_isobudget = scenario[0]
        seuil_VN_prospect_min = scenario[1]
        seuil_md = scenario[2]
        seuil_ret = scenario[3]
        seuil_SC = scenario[4]
        seuil_FV_inshore = scenario[5]
        seuil_satis = scenario[6]
        bool_capacitaire = scenario[7]
        print("seuil_ret : ", seuil_ret)
        print("seuil_sc : ", seuil_SC)
        verif_optimum = 1
        if verif_optimum == 1:
            print('----------Combinaison n.', combinaison, "    --------------")
            res_detaille = []
            res = pd.DataFrame()

            compteur = 0
            for mois in list_mois:
                print(mois)
                # print (VN_prev[compteur])
                # N_fiches_input[mois] = N_fiches_input[mois].astype(int)
                N_fiches = list(N_fiches_input[mois].astype(int))
                x = pulp.LpVariable.dicts("x", df2.index, lowBound=0, cat=pulp.LpContinuous)
                mod = pulp.LpProblem("Objectif", pulp.LpMaximize)

                # Objective function -> ce qu'on veut maximiser
                mod += sum([x[i] * df2.perf[i] * (df2.valeur[i] - df2.pu_booste[i]) for i in df2.index])

                # On veut distribuer toutes les fiches de chaque flux
                if bool_capacitaire == 1:
                    for k in range(len(flux)):
                        mod += sum([x[idx] for idx in range(k, len(centres) * len(flux), len(flux))]) == N_fiches[k]

                # Contrainte des 90 % pour le MD
                for k in range(len(flux)):
                    for idx in range(k, len(centres) * len(flux), len(flux)):
                        if df2.affectation[idx] == 1:
                            if 'MD' in df2.flux[idx]:
                                mod += x[idx] <= seuil_md * N_fiches[k]
                            elif 'ASR' in df2.flux[idx]:
                                mod += x[idx] <= seuil_ret * N_fiches[k]
                            elif df2.flux[idx] in list_SC_mono_site:
                                mod += x[idx] <= N_fiches
                            else:
                                mod += x[idx] <= seuil_SC * N_fiches[k]
                        else:
                            mod += x[idx] == 0

                # Contrainte des FV en inshore
                if N_fiches_FV.shape[0] != 0:
                    mod += sum([x[idx] for idx in list_FV_inshore]) == seuil_FV_inshore * N_fiches_FV[mois].sum()

                    # Ne pas depasser la capacite max de chaque centre d'appel
                for k in range(len(centres)):
                    mod += sum([x[idx] * df2.CA_FL[idx] for idx in range(k * len(flux), (k + 1) * len(flux))]) <= \
                           M_centres[
                               k]

                # contrainte du seuil de satisfaction a depasser
                if AE == 1:
                    mod += sum([x[idx] * df2.Satisfaction[idx] for idx in list_ae_index]) >= seuil_satis * sum(
                        fiches_ae[mois])

                # contrainte de l'isobudget par rapport au previsionnel de 2019
                print(list_budget_prev_2019[compteur])
                if bool_isobudget == 1:
                    mod += sum([x[idx] * df2.perf[idx] * df2.pu_booste[idx] for idx in range(df2.shape[0])]) <= \
                           list_budget_prev_2019[compteur]

                # contrainte du nombre de VN minimum pour le MD prospect
                if VN == 1:
                    if seuil_VN_prospect_min != 0:
                        mod += sum([x[idx] * df2.perf[idx] for idx in list_prospect_index]) >= seuil_VN_prospect_min * \
                               VN_prev[compteur]

                mod.solve()
                print(pulp.LpStatus[mod.status])
                if pulp.LpStatus[mod.status] != 'Optimal':
                    print('Optimal non trouvé pour ces paramètres')

                    scenarios_non_optimal.append(
                        {'bool_isobudget': bool_isobudget, 'seuil_VN_prospect_min': seuil_VN_prospect_min,
                         'seuil_md': seuil_md, 'seuil_ret': seuil_ret, 'seuil_SC': seuil_SC,
                         'seuil_FV_inshore': seuil_FV_inshore, 'seuil_satis': seuil_satis})
                    verif_optimum = 0
                compteur += 1
                if verif_optimum == 1:
                    res = pd.DataFrame({'centres': [df2.centres[idx] for idx in df2.index],
                                        'flux': [df2.flux[idx] for idx in df2.index],
                                        'nb_fiches': [int(x[idx].value()) for idx in df2.index]
                                        })

                    rendement = pd.merge(res, df2, how='left', on=['centres', 'flux'])
                    rendement["VN"] = rendement.nb_fiches * rendement.perf
                    rendement['CA'] = rendement.nb_fiches * rendement.CA_FL
                    rendement['valeur_final'] = rendement.nb_fiches * rendement.perf * (
                            rendement.valeur - rendement.pu_booste)
                    rendement['budget'] = rendement.VN * rendement.pu_booste
                    for row in rendement.itertuples():
                        if AE == 1 and AS == 1:
                            res_detaille.append(
                                {'Mois': mois, 'Centres': row.centres, 'Flux': row.flux, 'nb_fiches': row.nb_fiches,
                                 'perf': row.perf, 'valeur': row.valeur, 'CA_FL': row.CA_FL, 'PU': row.pu_booste,
                                 'VN': row.VN,
                                 'CA': row.CA, 'budget': row.budget, 'marge': row.valeur_final,
                                 'satis': row.Satisfaction})
                        if AE == 0 and AS == 1:  # on drop la colonne satisfaction qui n'existe pas
                            res_detaille.append(
                                {'Mois': mois, 'Centres': row.centres, 'Flux': row.flux, 'nb_fiches': row.nb_fiches,
                                 'perf': row.perf, 'valeur': row.valeur, 'CA_FL': row.CA_FL, 'PU': row.pu_booste,
                                 'VN': row.VN,
                                 'CA': row.CA, 'budget': row.budget, 'marge': row.valeur_final})

            if verif_optimum == 1:
                res_detaille = pd.DataFrame(res_detaille)

                res_detaille['activite'] = "SC"
                res_detaille.loc[res_detaille.Flux.str.contains("ASR_"), 'activite'] = "RET"
                res_detaille.loc[res_detaille.Flux.str.contains('FAI|Base|DESABONNES|PROSPECTS|Recents',
                                                                regex=True), 'activite'] = "Prospect"
                res_detaille.loc[res_detaille.Flux.str.contains("MIGRATION|ARPU|UPSELL|PROMO|Upgrade|Vente",
                                                                regex=True), 'activite'] = "Parc"

                res_detaille['budget'] = 0
                res_detaille['budget'] = res_detaille['PU'] * res_detaille['VN']

                res_detaille.loc[res_detaille.activite == "SC", 'marge'] = 0

                res_activite = res_detaille.groupby("activite").agg(
                    {"nb_fiches": "sum", "CA": "sum", "VN": "sum", "budget": "sum", "marge": "sum"}).reset_index()
                res_activite['seuil_md'] = seuil_md
                res_activite['seuil_satis'] = seuil_satis
                res_activite['seuil_ret'] = seuil_ret
                res_activite['seuil_SC'] = seuil_SC
                res_activite['seuil_FV_inshore'] = seuil_FV_inshore
                res_activite['seuil_VN_prospect_min'] = seuil_VN_prospect_min
                res_activite['isobudget'] = bool_isobudget
                res_activite['combinaison'] = combinaison
                for row in res_activite.itertuples():
                    res_detaille_final.append(
                        {'seuil_md': row.seuil_md, 'seuil_satis': row.seuil_satis, 'activite': row.activite,
                         'seuil_SC': row.seuil_SC, '#modele': row.combinaison,
                         'seuil_ret': row.seuil_ret, "seuil_FV_inshore": row.seuil_FV_inshore,
                         "seuil_VN_prospect_min": seuil_VN_prospect_min,
                         "isobudget": row.isobudget, 'nb_fiches': row.nb_fiches, 'CA': row.CA, 'VN': row.VN,
                         'budget': row.budget, 'marge': row.marge})
                combinaison += 1

    timestr = time.strftime("%Y%m%d_%H:%M")

    if len(scenarios_non_optimal) > 0:
        scenarios_non_optimal = pd.DataFrame(scenarios_non_optimal)
        scenarios_non_optimal = scenarios_non_optimal.drop_duplicates()
        scenarios_non_optimal.to_csv((output + 'scenarios_non_optim_' + timestr + '.csv').replace(":", "_"), sep=';', decimal=",",
                                     index=False)
        name_of_csv_noptim = (output + 'scenarios_non_optim_' + timestr + '.csv').replace(":", "_")

    if len(list_scenario) == 1:
        res_detaille.to_csv((output + 'scenario_detaille_' + timestr + '.csv').replace(":", "_"), sep=';', decimal=",",
                            index=False)

        name_of_csv = (output + 'scenario_detaille_' + timestr + '.csv').replace(":", "_")
    else:
        res_detaille_final = pd.DataFrame(res_detaille_final)
        res_detaille_final.to_csv((output + 'scenarios_' + timestr + '.csv').replace(":", "_"), sep=";", decimal=",", index=False)
        name_of_csv = (output + 'scenarios_' + timestr + '.csv').replace(":", "_")

    print('Exécution terminée!')

    return name_of_csv


if __name__=="__main__":
    optimization()
