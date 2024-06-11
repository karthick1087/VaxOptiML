import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
import joblib
import time

page = st.sidebar.radio("Choose a page", ["Home", "About", "Help"])
st.title("VaxOptiML")
st.write("An integrated pipeline designed to enhance cancer epitope prediction and prioritization")
st.image("logo.png", width=200)


text=['For epitope prediction in cancer immunotherapy, our integrated pipeline uses three models. We prioritize targets based on immunogenicity, precisely anticipate epitopes, and customize HLA pairing using sophisticated machine learning approaches and carefully selected datasets.',
       'Thorough assessment demonstrates improved performance compared to current methods, and graphic depictions highlight the effectiveness of our ensemble model in accelerating epitope discovery and vaccine development for cancer immunotherapy.',
       'After the input protein has been chunked, peptide features will be produced.',
       'On the basis of 0 and 1 annotations, peptides will be classified as either epitopes or non-epitopes.',
       'For such peptides, antigenic scores will be computed.Those peptides HLAs will be produced.',
       'Ultimately, the antigenic score and epitope nature will determine the final probable epitopes, as well as their starting and finishing places.']

text1=['Our integrated pipeline makes use of three models for epitope prediction in cancer immunotherapy. Using advanced machine learning techniques and carefully chosen datasets, we correctly predict epitopes, prioritize targets based on immunogenicity, and personalize HLA pairing.',
        'A thorough evaluation reveals enhanced performance when compared to current methods, and graphic representations illustrate the efficiency of our ensemble model in expediting epitope discovery and vaccine development for cancer immunotherapy.',
        'After chunking the input protein, peptide features will be generated.',
        'Peptides will be classed as epitopes or non-epitopes based on their 0 and 1 annotations.',
        'Antigenic scores for these peptides will be calculated.HLA peptides will be generated.',
        'Finally, the antigenic score and epitope nature will indicate the most likely epitopes, as well as their starting and ending positions.']

text2=['To anticipate epitopes for cancer immunotherapy, our integrated pipeline uses three models. By employing sophisticated machine learning methods and selective datasets, we are able to accurately forecast epitopes, rank targets according to immunogenicity, and customize HLA pairing.',
     'A detailed analysis shows improved performance over existing approaches, and visual aids demonstrate our ensemble models effectiveness in accelerating epitope discovery and vaccine development for cancer immunotherapy.',
       'Peptide features will be created following the chunking of the input protein.',
       'Peptide features will be created following the chunking of the input protein.Peptides 0 and 1 annotations will determine whether they are considered epitopes or non-epitopes.',
       'These peptides will have their antigenic scores determined.HLA peptides are going to be produced.',
       'Ultimately, the epitope nature and antigenic score will provide the most probable epitopes together with their beginning and terminating positions']

def main():
    if page == "Home":
        text_input = st.text_input("Enter Protein Sequence :").upper()
        text_input = text_input.replace(" ", "")  # This line removes spaces from the input sequence
        prediction_option = st.radio("Note- Remove the spaces from the amino acid sequnece if any:", ("MHC-1", "MHC-2", "BOTH"))
        if st.button("Predict"):
            if prediction_option == "MHC-1" and text_input:
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(6):
                    time.sleep(9)
                    progress_bar.progress((i + 1) * 100 // len(text))
                    status_text.text(f'**⏳ Analysis Initiated: {((i + 1) * 100 // len(text))}% ***')
                    st.write(f"[{i + 1}] ", text[i])
                status_text.text('# 💥 ** Analysis sucesssful - Output csv fles are loading....** 💥')

                protein_sequence = text_input
                def find_epitopes(sequence, window_size=10):
                    epitopes = []
                    start = []
                    end = []
                    for i in range(len(sequence) - window_size + 1):
                        epitope = sequence[i:i + window_size]
                        epitopes.append(epitope)
                        start.append(i)
                        end.append(i + window_size - 1)
                    return (epitopes, start, end)

                def is_valid_protein_sequence(peptide_sequence):
                    valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(peptide_sequence) <= valid_letters

                def calculate_atom_counts(peptide_sequence):
                    atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }
                    for aa in peptide_sequence:
                        aa = aa.upper()
                        if aa in aa_info:
                            atom_counts['H'] += aa_info[aa][0]
                            atom_counts['C'] += aa_info[aa][1]
                            atom_counts['N'] += aa_info[aa][2]
                            atom_counts['O'] += aa_info[aa][3]
                            atom_counts['S'] += aa_info[aa][4]

                    return atom_counts

                def calculate_physicochemical_properties(peptide_sequence):
                    if not is_valid_protein_sequence(peptide_sequence):
                        return [None] * 35
                    protein_analyzer = ProteinAnalysis(peptide_sequence)
                    theoretical_pI = protein_analyzer.isoelectric_point()
                    aliphatic_index = sum(kd[aa] for aa in peptide_sequence) / len(peptide_sequence)
                    positive_residues = sum(peptide_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    negative_residues = sum(peptide_sequence.count(aa) for aa in ['D', 'E'])
                    aromatic_count = protein_analyzer.aromaticity() * len(peptide_sequence)
                    polar_amino_acids = set("STNQ")
                    non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    polar_count = sum(peptide_sequence.count(aa) for aa in polar_amino_acids)
                    nonpolar_count = sum(peptide_sequence.count(aa) for aa in non_polar_amino_acids)
                    amino_acid_composition = protein_analyzer.get_amino_acids_percent()
                    molecular_weight = protein_analyzer.molecular_weight()
                    instability_index = protein_analyzer.instability_index()
                    aromaticity = protein_analyzer.aromaticity()
                    helix_fraction = protein_analyzer.secondary_structure_fraction()[0]
                    strand_fraction = protein_analyzer.secondary_structure_fraction()[1]
                    coil_fraction = protein_analyzer.secondary_structure_fraction()[2]
                    charge_at_pH_7 = protein_analyzer.charge_at_pH(7.0)
                    gravy = protein_analyzer.gravy()
                    amphipathicity = calculate_amphipathicity(peptide_sequence)
                    gravy_last_50 = protein_analyzer.gravy()
                    molar_extinction_coefficient = protein_analyzer.molar_extinction_coefficient()

                    return [theoretical_pI, aliphatic_index, positive_residues, negative_residues, aromatic_count,
                            polar_count, nonpolar_count, amino_acid_composition, molecular_weight, instability_index,
                            aromaticity, helix_fraction, strand_fraction, coil_fraction, charge_at_pH_7, gravy,
                            amphipathicity,
                            gravy_last_50, molar_extinction_coefficient]

                def calculate_amphipathicity(peptide_sequence):
                    hydrophobic_moment_scale = kd
                    hydrophobic_moment = sum(hydrophobic_moment_scale[aa] for aa in peptide_sequence)
                    mean_hydrophobicity = hydrophobic_moment / len(peptide_sequence)
                    return hydrophobic_moment - mean_hydrophobicity

                def process_single_protein(peptide_sequence, start, end):
                    atom_counts = calculate_atom_counts(peptide_sequence)
                    physicochemical_properties = calculate_physicochemical_properties(peptide_sequence)
                    total_atoms = sum(atom_counts.values())

                    result_dict = {'epitope': peptide_sequence,
                                   'start': start,
                                   'end': end,
                                   'H_Count': atom_counts['H'],
                                   'C_Count': atom_counts['C'],
                                   'N_Count': atom_counts['N'],
                                   'O_Count': atom_counts['O'],
                                   'S_Count': atom_counts['S'],
                                   'TotalAtoms_Count': total_atoms}

                    result_dict.update({
                        'Theoretical.pI': physicochemical_properties[0],
                        'Aliphatic.Index': physicochemical_properties[1],
                        'Positive.Residues': physicochemical_properties[2],
                        'Negative.Residues': physicochemical_properties[3],
                        'Aromatic.Count': physicochemical_properties[4],
                        'Polar.Count': physicochemical_properties[5],
                        'Nonpolar.Count': physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W',
                                   'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        result_dict[aa + '_Percent'] = physicochemical_properties[7].get(aa, 0)

                    result_dict.update({
                        'Molecular.Weight': physicochemical_properties[8],
                        'Instability.Index': physicochemical_properties[9],
                        'Aromaticity': physicochemical_properties[10],
                        'Helix.Fraction': physicochemical_properties[11],
                        'Strand.Fraction': physicochemical_properties[12],
                        'Coil.Fraction': physicochemical_properties[13],
                        'Charge.at.pH.7.0': physicochemical_properties[14],
                        'Gravy': physicochemical_properties[15],
                        'Amphipathicity': physicochemical_properties[16],
                        'GRAVY.Last.50': physicochemical_properties[17],
                        'Molar.Extinction.Coefficient': physicochemical_properties[18]
                    })

                    return result_dict

                def p_is_valid_protein_sequence(protein_sequence):
                    p_valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(protein_sequence) <= p_valid_letters

                def p_calculate_atom_counts(protein_sequence):
                    p_atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    p_aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }

                    for aa in protein_sequence:
                        aa = aa.upper()
                        if aa in p_aa_info:
                            p_atom_counts['H'] += p_aa_info[aa][0]
                            p_atom_counts['C'] += p_aa_info[aa][1]
                            p_atom_counts['N'] += p_aa_info[aa][2]
                            p_atom_counts['O'] += p_aa_info[aa][3]
                            p_atom_counts['S'] += p_aa_info[aa][4]

                    return p_atom_counts

                def p_calculate_physicochemical_properties(protein_sequence):
                    if not p_is_valid_protein_sequence(protein_sequence):
                        return [None] * 35

                    p_protein_analyzer = ProteinAnalysis(protein_sequence)

                    p_theoretical_pI = p_protein_analyzer.isoelectric_point()
                    p_aliphatic_index = sum(kd[aa] for aa in protein_sequence) / len(protein_sequence)
                    p_positive_residues = sum(protein_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    p_negative_residues = sum(protein_sequence.count(aa) for aa in ['D', 'E'])
                    p_aromatic_count = p_protein_analyzer.aromaticity() * len(protein_sequence)
                    p_polar_amino_acids = set("STNQ")
                    p_non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    p_polar_count = sum(protein_sequence.count(aa) for aa in p_polar_amino_acids)
                    p_nonpolar_count = sum(protein_sequence.count(aa) for aa in p_non_polar_amino_acids)
                    p_amino_acid_composition = p_protein_analyzer.get_amino_acids_percent()
                    p_molecular_weight = p_protein_analyzer.molecular_weight()
                    p_instability_index = p_protein_analyzer.instability_index()
                    p_aromaticity = p_protein_analyzer.aromaticity()
                    p_helix_fraction = p_protein_analyzer.secondary_structure_fraction()[0]
                    p_strand_fraction = p_protein_analyzer.secondary_structure_fraction()[1]
                    p_coil_fraction = p_protein_analyzer.secondary_structure_fraction()[2]
                    p_charge_at_pH_7 = p_protein_analyzer.charge_at_pH(7.0)
                    p_gravy = p_protein_analyzer.gravy()
                    p_amphipathicity = p_calculate_amphipathicity(protein_sequence)
                    p_gravy_last_50 = p_protein_analyzer.gravy()
                    p_molar_extinction_coefficient = p_protein_analyzer.molar_extinction_coefficient()

                    return [p_theoretical_pI, p_aliphatic_index, p_positive_residues, p_negative_residues,
                            p_aromatic_count,
                            p_polar_count, p_nonpolar_count, p_amino_acid_composition, p_molecular_weight,
                            p_instability_index,
                            p_aromaticity, p_helix_fraction, p_strand_fraction, p_coil_fraction, p_charge_at_pH_7,
                            p_gravy,
                            p_amphipathicity,
                            p_gravy_last_50, p_molar_extinction_coefficient]

                def p_calculate_amphipathicity(protein_sequence):
                    p_hydrophobic_moment_scale = kd
                    p_hydrophobic_moment = sum(p_hydrophobic_moment_scale[aa] for aa in protein_sequence)
                    p_mean_hydrophobicity = p_hydrophobic_moment / len(protein_sequence)
                    return p_hydrophobic_moment - p_mean_hydrophobicity

                def p_process_single_protein(protein_sequence):
                    p_atom_counts = p_calculate_atom_counts(protein_sequence)
                    p_physicochemical_properties = p_calculate_physicochemical_properties(protein_sequence)
                    p_total_atoms = sum(p_atom_counts.values())

                    p_result_dict = {'p_Sequence': protein_sequence,
                                     'p_H_Count': p_atom_counts['H'],
                                     'p_C_Count': p_atom_counts['C'],
                                     'p_N_Count': p_atom_counts['N'],
                                     'p_O_Count': p_atom_counts['O'],
                                     'p_S_Count': p_atom_counts['S'],
                                     'p_TotalAtoms_Count': p_total_atoms}

                    p_result_dict.update({
                        'p_Theoretical.pI': p_physicochemical_properties[0],
                        'p_Aliphatic.Index': p_physicochemical_properties[1],
                        'p_Positive.Residues': p_physicochemical_properties[2],
                        'p_Negative.Residues': p_physicochemical_properties[3],
                        'p_Aromatic.Count': p_physicochemical_properties[4],
                        'p_Polar.Count': p_physicochemical_properties[5],
                        'p_Nonpolar.Count': p_physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W', 'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        p_result_dict['p_' + aa + '_Percent'] = p_physicochemical_properties[7].get(aa, 0)

                    p_result_dict.update({
                        'p_Molecular.Weight': p_physicochemical_properties[8],
                        'p_Instability.Index': p_physicochemical_properties[9],
                        'p_Aromaticity': p_physicochemical_properties[10],
                        'p_Helix.Fraction': p_physicochemical_properties[11],
                        'p_Strand.Fraction': p_physicochemical_properties[12],
                        'p_Coil.Fraction': p_physicochemical_properties[13],
                        'p_Charge.at.pH.7.0': p_physicochemical_properties[14],
                        'p_Gravy': p_physicochemical_properties[15],
                        'p_Amphipathicity': p_physicochemical_properties[16],
                        'p_GRAVY.Last.50': p_physicochemical_properties[17],
                        'p_Molar.Extinction.Coefficient': p_physicochemical_properties[18]
                    })

                    return p_result_dict

                r_result = p_process_single_protein(protein_sequence)
                epitopes = find_epitopes(protein_sequence, window_size=10)
                epi = []
                for i in range(len(epitopes[0])):
                    result = process_single_protein(epitopes[0][i], epitopes[1][i], epitopes[2][i])
                    epi.append(result)

                df = pd.DataFrame(epi)
                file_name = 'epitopes_results.csv'
                df.to_csv(file_name)
                df_d = pd.read_csv(file_name)
                st.header("The epitope information")
                st.write(df_d)

                pro = []
                for i in range(len(epi)):
                    r_result = p_process_single_protein(protein_sequence)
                    pro.append(r_result)

                df_p = pd.DataFrame(pro)
                file_name = 'p_Sequence.csv'
                df_p.to_csv(file_name)
                df_d1 = pd.read_csv(file_name)
                st.header("The Protein sequence information")
                st.write(df_d1)

                df1 = pd.read_csv('epitopes_results.csv')
                df2 = pd.read_csv('p_Sequence.csv')
                merged_df = pd.merge(df1, df2, how='inner')
                merged_df.to_csv('result.csv', index=False)
                print("Merged CSV file has been created.")

                final_res = pd.read_csv('result.csv')
                st.header('The CSV with epitope information')
                st.write(final_res)

                inps = ['start', 'end', 'R_Percent', 'D_Percent', 'Q_Percent', 'H_Percent',
                        'I_Percent', 'L_Percent',
                        'K_Percent', 'S_Percent', 'Theoretical.pI',
                        'Aliphatic.Index', 'Helix.Fraction', 'Charge.at.pH.7.0', 'Amphipathicity',
                        'p_Molecular.Weight', 'p_Instability.Index', 'p_Helix.Fraction',
                        'p_Amphipathicity', 'p_Aliphatic.Index',
                        'p_H_Count', 'p_C_Count', 'p_N_Count',
                        'p_O_Count', 'p_S_Count', 'p_TotalAtoms_Count',
                        'p_A_Percent',
                        'p_D_Percent',
                        'p_E_Percent', 'p_G_Percent',
                        'p_I_Percent', 'p_K_Percent',
                        'p_F_Percent', 'p_T_Percent',
                        'p_V_Percent',
                        ]
                columns_to_extract = [final_res[j].values[:len(final_res)] for j in inps]
                columns_data = dict(zip(inps, columns_to_extract))
                columns_df = pd.DataFrame(columns_data)
                columns_df.to_csv('extracted_columns.csv')
                bagging_pred = []
                extra_trees_pred = []
                random_forest_pred = []
                df = pd.read_csv('extracted_columns.csv')
                print(df.columns)
                st.header("The extracted Columns")
                st.write(df)
                for i in range(len(df)):
                    print(df.end.values[i])
                    print(
                        '-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print(f'FOR THE {final_res.epitope[i]} the value if 1-> epitope and o-> non-epitope')
                    print(
                        '------------------------------------------------------------------------------------------------------------')

                    inp = [df.start.values[i], df.end.values[i], df.R_Percent.values[i], df.D_Percent.values[i],
                           df.Q_Percent.values[i], df.H_Percent.values[i],
                           df.I_Percent.values[i], df.L_Percent.values[i], df.K_Percent.values[i],
                           df.S_Percent.values[i], df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                           df['Helix.Fraction'].values[i], df['Charge.at.pH.7.0'].values[i],
                           df['Amphipathicity'].values[i],
                           df['p_Molecular.Weight'].values[i], df['p_Instability.Index'].values[i],
                           df['p_Helix.Fraction'].values[i],
                           df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                           df['p_H_Count'].values[i],
                           df['p_C_Count'].values[i], df['p_N_Count'].values[i], df['p_O_Count'].values[i],
                           df['p_S_Count'].values[i], df['p_TotalAtoms_Count'].values[i], df['p_A_Percent'].values[i],
                           df['p_D_Percent'].values[i], df['p_E_Percent'].values[i], df['p_G_Percent'].values[i],
                           df['p_I_Percent'].values[i], df['p_K_Percent'].values[i], df['p_F_Percent'].values[i],
                           df['p_T_Percent'].values[i], df['p_V_Percent'].values[i]]

                    bagging = joblib.load('Bagging_tar_mhc1.pkl')
                    pred_bag = bagging.predict([inp])
                    bagging_pred.append(pred_bag[0])
                    print("The prediction using Bagging ", pred_bag)

                    extratree = joblib.load('extratree_tar_mhc1.pkl')
                    predict = extratree.predict([inp])
                    extra_trees_pred.append(predict[0])
                    print("The extra tree prediction ", predict)

                    randomfor = joblib.load('randomforest_tar_mhc1.pkl')
                    random_pred = randomfor.predict([inp])
                    random_forest_pred.append(random_pred[0])
                    print("The random forest ", random_pred)

                df = pd.read_csv('result.csv')
                print(df.columns)
                epitopes = []
                xg_boost = []
                lgbm_score = []
                start_val = []
                end_val = []

                for i in range(len(df)):
                    print('-------------------------------------------')
                    epitopes.append(df.epitope.values[i])
                    start_val.append(df.start.values[i])
                    end_val.append(df.end.values[i])
                    print("---------------------------------------------")
                    print(df.epitope.values[i])

                    print('---------------------------------------------')
                    score_inp = [df.start.values[i], df.end.values[i],
                                 df['A_Percent'].values[i], df['R_Percent'].values[i],
                                 df['N_Percent'].values[i], df['D_Percent'].values[i],
                                 df['C_Percent'].values[i], df['E_Percent'].values[i],
                                 df['Q_Percent'].values[i], df['G_Percent'].values[i], df['H_Percent'].values[i],
                                 df['I_Percent'].values[i], df['L_Percent'].values[i], df['K_Percent'].values[i],
                                 df['M_Percent'].values[i], df['F_Percent'].values[i], df['P_Percent'].values[i],
                                 df['S_Percent'].values[i], df['T_Percent'].values[i], df['W_Percent'].values[i],
                                 df['Y_Percent'].values[i], df['V_Percent'].values[i],
                                 df['H_Count'].values[i], df['C_Count'].values[i], df['N_Count'].values[i],
                                 df['S_Count'].values[i],
                                 df['TotalAtoms_Count'].values[i],
                                 df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                                 df['Positive.Residues'].values[i], df['Negative.Residues'].values[i],
                                 df['Aromatic.Count'].values[i], df['Polar.Count'].values[i],
                                 df['Nonpolar.Count'].values[i], df['Molecular.Weight'].values[i],
                                 df['Instability.Index'].values[i], df['Aromaticity'].values[i],
                                 df['Helix.Fraction'].values[i], df['Strand.Fraction'].values[i],
                                 df['Coil.Fraction'].values[i],
                                 df['Charge.at.pH.7.0'].values[i], df['Amphipathicity'].values[i],
                                 df['GRAVY.Last.50'].values[i],
                                 df['p_Instability.Index'].values[i], df['p_Helix.Fraction'].values[i],
                                 df['p_Strand.Fraction'].values[i], df['p_Coil.Fraction'].values[i],
                                 df['p_Charge.at.pH.7.0'].values[i],
                                 df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                                 df['p_Aromatic.Count'].values[i], df['p_Nonpolar.Count'].values[i],
                                 df['p_H_Count'].values[i],
                                 df['p_C_Count'].values[i], df['p_O_Count'].values[i],
                                 df['p_TotalAtoms_Count'].values[i],
                                 df['p_R_Percent'].values[i],
                                 df['p_N_Percent'].values[i], df['p_D_Percent'].values[i],
                                 df['p_E_Percent'].values[i],
                                 df['p_L_Percent'].values[i],
                                 df['p_T_Percent'].values[i], df['p_W_Percent'].values[i]]

                    xgb = joblib.load('xgb_score_mhc1.pkl')
                    xgb_pred = xgb.predict([score_inp])
                    print("The xgb_pred ", xgb_pred)
                    xg_boost.append(xgb_pred[0])

                    lgb = joblib.load('lgb_score_mhc1.pkl')
                    lgbm_prediction = lgb.predict([score_inp])
                    lgbm_score.append(lgbm_prediction[0])
                    print('The lgbm prediction ', lgbm_prediction)

                kolaskar_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                })

                kolaskar_df.to_csv('kolaskar.csv')
                df_kolaskar = pd.read_csv("kolaskar.csv")
                st.header('The Kolaskar score information')
                st.write(df_kolaskar)

                def prot_to_num(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }
                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) + float(numeric_multiplier)
                    return total_score

                def protein_numerical(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                protein_sequence = "AAAALNGVDRRSLQRSARLALEVLERAKRRAVDWHALERPKGCMGVLAREAPHLEKQPAAGPQRVLPGEKYYSSVPEEGGATHVYRYHRGESKLHMCLDIGNGQAENISKDLYIEVYPGTYSVTVGSNDLTKKTHVVAVDSGQSVDLVFPV"
                df_hla = pd.read_csv('result.csv')
                ext_hla = []
                lgbm_hla = []
                hist_hla = []
                for i in range(len(df_hla)):
                    print(f"The HLA Prediction for {df_hla.epitope.values[i]}")
                    hla_inp = [extra_trees_pred[i], df_hla['C_Percent'].values[i], df_hla['Q_Percent'].values[i],
                               df_hla['G_Percent'].values[i],
                               df_hla['K_Percent'].values[i], df_hla['P_Percent'].values[i],
                               df_hla['S_Percent'].values[i],
                               df_hla['T_Percent'].values[i], df_hla['W_Percent'].values[i],
                               df_hla['H_Count'].values[i],
                               df_hla['C_Count'].values[i],
                               df_hla['N_Count'].values[i], df_hla['O_Count'].values[i],
                               df_hla['TotalAtoms_Count'].values[i],
                               df_hla['Theoretical.pI'].values[i],
                               df_hla['Positive.Residues'].values[i], df_hla['Negative.Residues'].values[i],
                               df_hla['Aromatic.Count'].values[i], df_hla['Polar.Count'].values[i],
                               df_hla['Nonpolar.Count'].values[i], df_hla['Molecular.Weight'].values[i],
                               df_hla['Instability.Index'].values[i], df_hla['Strand.Fraction'].values[i],
                               df_hla['Charge.at.pH.7.0'].values[i], df_hla['p_Aromaticity'].values[i],
                               df_hla['p_Strand.Fraction'].values[i], df_hla['p_Coil.Fraction'].values[i],
                               df_hla['p_Gravy'].values[i], df_hla['p_Amphipathicity'].values[i],
                               df_hla['p_GRAVY.Last.50'].values[i],
                               df_hla['p_Aliphatic.Index'].values[i], df_hla['p_Polar.Count'].values[i],
                               df_hla['p_N_Percent'].values[i], df_hla['p_C_Percent'].values[i],
                               df_hla['p_K_Percent'].values[i], df_hla['p_F_Percent'].values[i],
                               df_hla['p_P_Percent'].values[i], df_hla['p_S_Percent'].values[i],
                               df_hla['p_T_Percent'].values[i], df_hla['p_W_Percent'].values[i],
                               df_hla['p_V_Percent'].values[i], 0, protein_numerical(text_input)]
                    ext = joblib.load('extra_tree_hla_mhc1.pkl')
                    pred = ext.predict([hla_inp])[0]
                    ext_hla.append(pred)
                    print("The extra trees hla is ", pred)

                    lgbm = joblib.load('xgbr_hla_mhc1.pkl')
                    lgbm_hla.append(lgbm.predict([hla_inp])[0])
                    print("The LGBM prediction ", lgbm.predict([hla_inp])[0])


                    hist = joblib.load('hist_hla_mhc1.pkl')
                    hist_hla.append(hist.predict([hla_inp])[0])

                    print("The Hist_prediction ", hist.predict([hla_inp])[0])

                score_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                    "lgbm_prediction": lgbm_hla,
                    "extra_hla": ext_hla,
                    "hist_hla": hist_hla
                })

                score_df.to_csv("final_output.csv")
                score = pd.read_csv('final_output.csv')
                st.header("The File with score and values")
                st.write(score)

                print(score['Random_forest_Target'].values)
                print(score['Extra_tree_Target'].values)
                count_ones = score[['Extra_tree_Target', 'Random_forest_Target', 'bagging_Target']].sum(axis=1)
                score['Target'] = (count_ones > 2).astype(int)
                score.to_csv('target_final.csv')
                df_final = pd.read_csv('target_final.csv')
                df = df_final['extra_hla']
                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }

                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) * float(numeric_multiplier)
                    return total_score

                def find_nearest_hla(total_score, hla_strings, k=10):
                    nearest_hla_list = []
                    hla_strings_sorted = sorted(hla_strings,
                                                key=lambda hla: abs(calculate_numerical_score(hla) - total_score))
                    for hla in hla_strings_sorted[:k]:
                        nearest_hla_list.append(hla)
                    return nearest_hla_list
                data=pd.read_csv('hla.csv')
                hla_output = []
                pred = []
                hla_strings = [str(hla) for hla in data['HLA'].values]
                for i in range(len(df_final)):
                    nearest_hla_list = find_nearest_hla(df_final['extra_hla'].values[i], hla_strings, k=10)
                    print("10 Nearest HLA Strings:")
                    for i, hla in enumerate(nearest_hla_list, start=1):
                        print(hla)
                        pred.append(hla)
                        if len(pred) == 10:
                            hla_output.append(pred)
                            pred = []
                df_final['hla_values'] = hla_output
                df_final.to_csv("expected.csv")
                df_d3 = pd.read_csv("expected.csv")
                target = df_d3[['Extra_tree_Target', 'Random_forest_Target']].sum(axis=1)
                print('---------------------------------------------')
                print(target)
                df_d3['Target'] = (target > 1).astype(int)
                print(df_d3.Target)
                print('-------------------------------------------------------')
                df_tar = df_d3[df_d3['Target'] == 1]
                if len(df_tar.Target.values) <= 30:
                    df_tar.to_csv('target.csv')
                    df_tab = pd.read_csv('target.csv')
                    print(df_tab.columns)
                    col = ['start', 'end', 'Epitope', 'hla_values']
                    st.table(df_tab[col])
                else:
                    values = df_d3['XGB_predicted_score'].sort_values(ascending=False).values
                    val = []
                    for i in range(len(values)):
                        val.append(values[i])
                        if len(val) == 30:
                            break
                    epitope = []
                    hla = []
                    starts = []
                    ends = []
                    for i in val:
                        df_val = df_d3[df_d3['XGB_predicted_score'] == i]
                        epitope.append(df_val.Epitope)
                        hla.append(df_val.hla_values)
                        starts.append(df_val.start)
                        ends.append(df_val.end)
                    data_dict = {
                        'Epitope': epitope,
                        'HLA': hla,
                        'Start': starts,
                        'End': ends
                    }
                    df = pd.DataFrame(data_dict)
                    df = df.explode('Epitope').explode('HLA').explode('Start').explode('End')
                    df.reset_index(drop=True, inplace=True)
                    print(df)
                    st.write(df)
            elif prediction_option== "MHC-2" and text_input:
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(6):
                    time.sleep(9)
                    progress_bar.progress((i + 1) * 100 // len(text1))
                    status_text.text(f'**⏳ Countdown: {((i + 1) * 100 // len(text1))}% ***')
                    st.write(f"[{i + 1}] ", text1[i])
                status_text.text('# 💥 **Boom!** 💥')

                protein_sequence = text_input
                def find_epitopes(sequence, window_size=15):
                    epitopes = []
                    start = []
                    end = []
                    for i in range(len(sequence) - window_size + 1):
                        epitope = sequence[i:i + window_size]
                        epitopes.append(epitope)
                        start.append(i)
                        end.append(i + window_size - 1)
                    return (epitopes, start, end)

                def is_valid_protein_sequence(peptide_sequence):
                    valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(peptide_sequence) <= valid_letters

                def calculate_atom_counts(peptide_sequence):
                    atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }
                    for aa in peptide_sequence:
                        aa = aa.upper()
                        if aa in aa_info:
                            atom_counts['H'] += aa_info[aa][0]
                            atom_counts['C'] += aa_info[aa][1]
                            atom_counts['N'] += aa_info[aa][2]
                            atom_counts['O'] += aa_info[aa][3]
                            atom_counts['S'] += aa_info[aa][4]

                    return atom_counts

                def calculate_physicochemical_properties(peptide_sequence):
                    if not is_valid_protein_sequence(peptide_sequence):
                        return [None] * 35
                    protein_analyzer = ProteinAnalysis(peptide_sequence)
                    theoretical_pI = protein_analyzer.isoelectric_point()
                    aliphatic_index = sum(kd[aa] for aa in peptide_sequence) / len(peptide_sequence)
                    positive_residues = sum(peptide_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    negative_residues = sum(peptide_sequence.count(aa) for aa in ['D', 'E'])
                    aromatic_count = protein_analyzer.aromaticity() * len(peptide_sequence)
                    polar_amino_acids = set("STNQ")
                    non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    polar_count = sum(peptide_sequence.count(aa) for aa in polar_amino_acids)
                    nonpolar_count = sum(peptide_sequence.count(aa) for aa in non_polar_amino_acids)
                    amino_acid_composition = protein_analyzer.get_amino_acids_percent()
                    molecular_weight = protein_analyzer.molecular_weight()
                    instability_index = protein_analyzer.instability_index()
                    aromaticity = protein_analyzer.aromaticity()
                    helix_fraction = protein_analyzer.secondary_structure_fraction()[0]
                    strand_fraction = protein_analyzer.secondary_structure_fraction()[1]
                    coil_fraction = protein_analyzer.secondary_structure_fraction()[2]
                    charge_at_pH_7 = protein_analyzer.charge_at_pH(7.0)
                    gravy = protein_analyzer.gravy()
                    amphipathicity = calculate_amphipathicity(peptide_sequence)
                    gravy_last_50 = protein_analyzer.gravy()
                    molar_extinction_coefficient = protein_analyzer.molar_extinction_coefficient()

                    return [theoretical_pI, aliphatic_index, positive_residues, negative_residues, aromatic_count,
                            polar_count, nonpolar_count, amino_acid_composition, molecular_weight, instability_index,
                            aromaticity, helix_fraction, strand_fraction, coil_fraction, charge_at_pH_7, gravy,
                            amphipathicity,
                            gravy_last_50, molar_extinction_coefficient]

                def calculate_amphipathicity(peptide_sequence):
                    hydrophobic_moment_scale = kd
                    hydrophobic_moment = sum(hydrophobic_moment_scale[aa] for aa in peptide_sequence)
                    mean_hydrophobicity = hydrophobic_moment / len(peptide_sequence)
                    return hydrophobic_moment - mean_hydrophobicity

                def process_single_protein(peptide_sequence, start, end):
                    atom_counts = calculate_atom_counts(peptide_sequence)
                    physicochemical_properties = calculate_physicochemical_properties(peptide_sequence)
                    total_atoms = sum(atom_counts.values())

                    result_dict = {'epitope': peptide_sequence,
                                   'start': start,
                                   'end': end,
                                   'H_Count': atom_counts['H'],
                                   'C_Count': atom_counts['C'],
                                   'N_Count': atom_counts['N'],
                                   'O_Count': atom_counts['O'],
                                   'S_Count': atom_counts['S'],
                                   'TotalAtoms_Count': total_atoms}

                    result_dict.update({
                        'Theoretical.pI': physicochemical_properties[0],
                        'Aliphatic.Index': physicochemical_properties[1],
                        'Positive.Residues': physicochemical_properties[2],
                        'Negative.Residues': physicochemical_properties[3],
                        'Aromatic.Count': physicochemical_properties[4],
                        'Polar.Count': physicochemical_properties[5],
                        'Nonpolar.Count': physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W',
                                   'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        result_dict[aa + '_Percent'] = physicochemical_properties[7].get(aa, 0)

                    result_dict.update({
                        'Molecular.Weight': physicochemical_properties[8],
                        'Instability.Index': physicochemical_properties[9],
                        'Aromaticity': physicochemical_properties[10],
                        'Helix.Fraction': physicochemical_properties[11],
                        'Strand.Fraction': physicochemical_properties[12],
                        'Coil.Fraction': physicochemical_properties[13],
                        'Charge.at.pH.7.0': physicochemical_properties[14],
                        'Gravy': physicochemical_properties[15],
                        'Amphipathicity': physicochemical_properties[16],
                        'GRAVY.Last.50': physicochemical_properties[17],
                        'Molar.Extinction.Coefficient': physicochemical_properties[18]
                    })

                    return result_dict

                def p_is_valid_protein_sequence(protein_sequence):
                    p_valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(protein_sequence) <= p_valid_letters

                def p_calculate_atom_counts(protein_sequence):
                    p_atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    p_aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }

                    for aa in protein_sequence:
                        aa = aa.upper()
                        if aa in p_aa_info:
                            p_atom_counts['H'] += p_aa_info[aa][0]
                            p_atom_counts['C'] += p_aa_info[aa][1]
                            p_atom_counts['N'] += p_aa_info[aa][2]
                            p_atom_counts['O'] += p_aa_info[aa][3]
                            p_atom_counts['S'] += p_aa_info[aa][4]

                    return p_atom_counts

                def p_calculate_physicochemical_properties(protein_sequence):
                    if not p_is_valid_protein_sequence(protein_sequence):
                        return [None] * 35

                    p_protein_analyzer = ProteinAnalysis(protein_sequence)

                    p_theoretical_pI = p_protein_analyzer.isoelectric_point()
                    p_aliphatic_index = sum(kd[aa] for aa in protein_sequence) / len(protein_sequence)
                    p_positive_residues = sum(protein_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    p_negative_residues = sum(protein_sequence.count(aa) for aa in ['D', 'E'])
                    p_aromatic_count = p_protein_analyzer.aromaticity() * len(protein_sequence)
                    p_polar_amino_acids = set("STNQ")
                    p_non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    p_polar_count = sum(protein_sequence.count(aa) for aa in p_polar_amino_acids)
                    p_nonpolar_count = sum(protein_sequence.count(aa) for aa in p_non_polar_amino_acids)
                    p_amino_acid_composition = p_protein_analyzer.get_amino_acids_percent()
                    p_molecular_weight = p_protein_analyzer.molecular_weight()
                    p_instability_index = p_protein_analyzer.instability_index()
                    p_aromaticity = p_protein_analyzer.aromaticity()
                    p_helix_fraction = p_protein_analyzer.secondary_structure_fraction()[0]
                    p_strand_fraction = p_protein_analyzer.secondary_structure_fraction()[1]
                    p_coil_fraction = p_protein_analyzer.secondary_structure_fraction()[2]
                    p_charge_at_pH_7 = p_protein_analyzer.charge_at_pH(7.0)
                    p_gravy = p_protein_analyzer.gravy()
                    p_amphipathicity = p_calculate_amphipathicity(protein_sequence)
                    p_gravy_last_50 = p_protein_analyzer.gravy()
                    p_molar_extinction_coefficient = p_protein_analyzer.molar_extinction_coefficient()

                    return [p_theoretical_pI, p_aliphatic_index, p_positive_residues, p_negative_residues,
                            p_aromatic_count,
                            p_polar_count, p_nonpolar_count, p_amino_acid_composition, p_molecular_weight,
                            p_instability_index,
                            p_aromaticity, p_helix_fraction, p_strand_fraction, p_coil_fraction, p_charge_at_pH_7,
                            p_gravy,
                            p_amphipathicity,
                            p_gravy_last_50, p_molar_extinction_coefficient]

                def p_calculate_amphipathicity(protein_sequence):
                    p_hydrophobic_moment_scale = kd
                    p_hydrophobic_moment = sum(p_hydrophobic_moment_scale[aa] for aa in protein_sequence)
                    p_mean_hydrophobicity = p_hydrophobic_moment / len(protein_sequence)
                    return p_hydrophobic_moment - p_mean_hydrophobicity

                def p_process_single_protein(protein_sequence):
                    p_atom_counts = p_calculate_atom_counts(protein_sequence)
                    p_physicochemical_properties = p_calculate_physicochemical_properties(protein_sequence)
                    p_total_atoms = sum(p_atom_counts.values())

                    p_result_dict = {'p_Sequence': protein_sequence,
                                     'p_H_Count': p_atom_counts['H'],
                                     'p_C_Count': p_atom_counts['C'],
                                     'p_N_Count': p_atom_counts['N'],
                                     'p_O_Count': p_atom_counts['O'],
                                     'p_S_Count': p_atom_counts['S'],
                                     'p_TotalAtoms_Count': p_total_atoms}

                    p_result_dict.update({
                        'p_Theoretical.pI': p_physicochemical_properties[0],
                        'p_Aliphatic.Index': p_physicochemical_properties[1],
                        'p_Positive.Residues': p_physicochemical_properties[2],
                        'p_Negative.Residues': p_physicochemical_properties[3],
                        'p_Aromatic.Count': p_physicochemical_properties[4],
                        'p_Polar.Count': p_physicochemical_properties[5],
                        'p_Nonpolar.Count': p_physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W', 'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        p_result_dict['p_' + aa + '_Percent'] = p_physicochemical_properties[7].get(aa, 0)

                    p_result_dict.update({
                        'p_Molecular.Weight': p_physicochemical_properties[8],
                        'p_Instability.Index': p_physicochemical_properties[9],
                        'p_Aromaticity': p_physicochemical_properties[10],
                        'p_Helix.Fraction': p_physicochemical_properties[11],
                        'p_Strand.Fraction': p_physicochemical_properties[12],
                        'p_Coil.Fraction': p_physicochemical_properties[13],
                        'p_Charge.at.pH.7.0': p_physicochemical_properties[14],
                        'p_Gravy': p_physicochemical_properties[15],
                        'p_Amphipathicity': p_physicochemical_properties[16],
                        'p_GRAVY.Last.50': p_physicochemical_properties[17],
                        'p_Molar.Extinction.Coefficient': p_physicochemical_properties[18]
                    })

                    return p_result_dict

                r_result = p_process_single_protein(protein_sequence)
                epitopes = find_epitopes(protein_sequence, window_size=15)
                epi = []
                for i in range(len(epitopes[0])):
                    result = process_single_protein(epitopes[0][i], epitopes[1][i], epitopes[2][i])
                    epi.append(result)

                df = pd.DataFrame(epi)
                file_name = 'epitopes_results.csv'
                df.to_csv(file_name)
                df_d = pd.read_csv(file_name)
                st.header("The epitope information")
                st.write(df_d)

                pro = []
                for i in range(len(epi)):
                    r_result = p_process_single_protein(protein_sequence)
                    pro.append(r_result)

                df_p = pd.DataFrame(pro)
                file_name = 'p_Sequence.csv'
                df_p.to_csv(file_name)
                df_d1 = pd.read_csv(file_name)
                st.header("The Protein sequence information")
                st.write(df_d1)

                df1 = pd.read_csv('epitopes_results.csv')
                df2 = pd.read_csv('p_Sequence.csv')
                merged_df = pd.merge(df1, df2, how='inner')
                merged_df.to_csv('result.csv', index=False)
                print("Merged CSV file has been created.")

                final_res = pd.read_csv('result.csv')
                st.header('The CSV with epitope information')
                st.write(final_res)

                inps = ['start', 'end', 'R_Percent', 'D_Percent', 'Q_Percent', 'H_Percent',
                        'I_Percent', 'L_Percent',
                        'K_Percent', 'S_Percent', 'Theoretical.pI',
                        'Aliphatic.Index', 'Helix.Fraction', 'Charge.at.pH.7.0', 'Amphipathicity',
                        'p_Molecular.Weight', 'p_Instability.Index', 'p_Helix.Fraction',
                        'p_Amphipathicity', 'p_Aliphatic.Index',
                        'p_H_Count', 'p_C_Count', 'p_N_Count',
                        'p_O_Count', 'p_S_Count', 'p_TotalAtoms_Count',
                        'p_A_Percent',
                        'p_D_Percent',
                        'p_E_Percent', 'p_G_Percent',
                        'p_I_Percent', 'p_K_Percent',
                        'p_F_Percent', 'p_T_Percent',
                        'p_V_Percent',
                        ]
                columns_to_extract = [final_res[j].values[:len(final_res)] for j in inps]
                columns_data = dict(zip(inps, columns_to_extract))
                columns_df = pd.DataFrame(columns_data)
                columns_df.to_csv('extracted_columns.csv')
                bagging_pred = []
                extra_trees_pred = []
                random_forest_pred = []
                df = pd.read_csv('extracted_columns.csv')
                print(df.columns)
                st.header("The extracted Columns")
                st.write(df)
                for i in range(len(df)):
                    print(df.end.values[i])
                    print(
                        '-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print(f'FOR THE {final_res.epitope[i]} the value if 1-> epitope and o-> non-epitope')
                    print(
                        '------------------------------------------------------------------------------------------------------------')

                    inp = [df.start.values[i], df.end.values[i], df.R_Percent.values[i], df.D_Percent.values[i],
                           df.Q_Percent.values[i], df.H_Percent.values[i],
                           df.I_Percent.values[i], df.L_Percent.values[i], df.K_Percent.values[i],
                           df.S_Percent.values[i], df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                           df['Helix.Fraction'].values[i], df['Charge.at.pH.7.0'].values[i],
                           df['Amphipathicity'].values[i],
                           df['p_Molecular.Weight'].values[i], df['p_Instability.Index'].values[i],
                           df['p_Helix.Fraction'].values[i],
                           df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                           df['p_H_Count'].values[i],
                           df['p_C_Count'].values[i], df['p_N_Count'].values[i], df['p_O_Count'].values[i],
                           df['p_S_Count'].values[i], df['p_TotalAtoms_Count'].values[i], df['p_A_Percent'].values[i],
                           df['p_D_Percent'].values[i], df['p_E_Percent'].values[i], df['p_G_Percent'].values[i],
                           df['p_I_Percent'].values[i], df['p_K_Percent'].values[i], df['p_F_Percent'].values[i],
                           df['p_T_Percent'].values[i], df['p_V_Percent'].values[i]]

                    bagging = joblib.load('Bagging_tar_mhc1.pkl')
                    pred_bag = bagging.predict([inp])
                    bagging_pred.append(pred_bag[0])
                    print("The prediction using Bagging ", pred_bag)

                    extratree = joblib.load('extratree_tar_mhc1.pkl')
                    predict = extratree.predict([inp])
                    extra_trees_pred.append(predict[0])
                    print("The extra tree prediction ", predict)

                    randomfor = joblib.load('randomforest_tar_mhc1.pkl')
                    random_pred = randomfor.predict([inp])
                    random_forest_pred.append(random_pred[0])
                    print("The random forest ", random_pred)

                df = pd.read_csv('result.csv')
                print(df.columns)
                epitopes = []
                xg_boost = []
                lgbm_score = []
                start_val = []
                end_val = []

                for i in range(len(df)):
                    print('-------------------------------------------')
                    epitopes.append(df.epitope.values[i])
                    start_val.append(df.start.values[i])
                    end_val.append(df.end.values[i])
                    print("---------------------------------------------")
                    print(df.epitope.values[i])

                    print('---------------------------------------------')
                    score_inp = [df.start.values[i], df.end.values[i],
                                 df['A_Percent'].values[i], df['R_Percent'].values[i],
                                 df['N_Percent'].values[i], df['D_Percent'].values[i],
                                 df['C_Percent'].values[i], df['E_Percent'].values[i],
                                 df['Q_Percent'].values[i], df['G_Percent'].values[i], df['H_Percent'].values[i],
                                 df['I_Percent'].values[i], df['L_Percent'].values[i], df['K_Percent'].values[i],
                                 df['M_Percent'].values[i], df['F_Percent'].values[i], df['P_Percent'].values[i],
                                 df['S_Percent'].values[i], df['T_Percent'].values[i], df['W_Percent'].values[i],
                                 df['Y_Percent'].values[i], df['V_Percent'].values[i],
                                 df['H_Count'].values[i], df['C_Count'].values[i], df['N_Count'].values[i],
                                 df['S_Count'].values[i],
                                 df['TotalAtoms_Count'].values[i],
                                 df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                                 df['Positive.Residues'].values[i], df['Negative.Residues'].values[i],
                                 df['Aromatic.Count'].values[i], df['Polar.Count'].values[i],
                                 df['Nonpolar.Count'].values[i], df['Molecular.Weight'].values[i],
                                 df['Instability.Index'].values[i], df['Aromaticity'].values[i],
                                 df['Helix.Fraction'].values[i], df['Strand.Fraction'].values[i],
                                 df['Coil.Fraction'].values[i],
                                 df['Charge.at.pH.7.0'].values[i], df['Amphipathicity'].values[i],
                                 df['GRAVY.Last.50'].values[i],
                                 df['p_Instability.Index'].values[i], df['p_Helix.Fraction'].values[i],
                                 df['p_Strand.Fraction'].values[i], df['p_Coil.Fraction'].values[i],
                                 df['p_Charge.at.pH.7.0'].values[i],
                                 df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                                 df['p_Aromatic.Count'].values[i], df['p_Nonpolar.Count'].values[i],
                                 df['p_H_Count'].values[i],
                                 df['p_C_Count'].values[i], df['p_O_Count'].values[i],
                                 df['p_TotalAtoms_Count'].values[i],
                                 df['p_R_Percent'].values[i],
                                 df['p_N_Percent'].values[i], df['p_D_Percent'].values[i],
                                 df['p_E_Percent'].values[i],
                                 df['p_L_Percent'].values[i],
                                 df['p_T_Percent'].values[i], df['p_W_Percent'].values[i]]

                    xgb = joblib.load('xgb_score_mhc1.pkl')
                    xgb_pred = xgb.predict([score_inp])
                    print("The xgb_pred ", xgb_pred)
                    xg_boost.append(xgb_pred[0])

                    lgb = joblib.load('lgb_score_mhc1.pkl')
                    lgbm_prediction = lgb.predict([score_inp])
                    lgbm_score.append(lgbm_prediction[0])
                    print('The lgbm prediction ', lgbm_prediction)

                kolaskar_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                })

                kolaskar_df.to_csv('kolaskar.csv')
                df_kolaskar = pd.read_csv("kolaskar.csv")
                st.header('The Kolaskar score information')
                st.write(df_kolaskar)

                def prot_to_num(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }

                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) + float(numeric_multiplier)
                    return total_score

                def protein_numerical(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                protein_sequence = "AAAALNGVDRRSLQRSARLALEVLERAKRRAVDWHALERPKGCMGVLAREAPHLEKQPAAGPQRVLPGEKYYSSVPEEGGATHVYRYHRGESKLHMCLDIGNGQAENISKDLYIEVYPGTYSVTVGSNDLTKKTHVVAVDSGQSVDLVFPV"
                df_hla = pd.read_csv('result.csv')
                ext_hla = []
                lgbm_hla = []
                hist_hla = []
                for i in range(len(df_hla)):
                    print(f"The HLA Prediction for {df_hla.epitope.values[i]}")
                    hla_inp = [extra_trees_pred[i], df_hla['C_Percent'].values[i], df_hla['Q_Percent'].values[i],
                               df_hla['G_Percent'].values[i],
                               df_hla['K_Percent'].values[i], df_hla['P_Percent'].values[i],
                               df_hla['S_Percent'].values[i],
                               df_hla['T_Percent'].values[i], df_hla['W_Percent'].values[i],
                               df_hla['H_Count'].values[i],
                               df_hla['C_Count'].values[i],
                               df_hla['N_Count'].values[i], df_hla['O_Count'].values[i],
                               df_hla['TotalAtoms_Count'].values[i],
                               df_hla['Theoretical.pI'].values[i],
                               df_hla['Positive.Residues'].values[i], df_hla['Negative.Residues'].values[i],
                               df_hla['Aromatic.Count'].values[i], df_hla['Polar.Count'].values[i],
                               df_hla['Nonpolar.Count'].values[i], df_hla['Molecular.Weight'].values[i],
                               df_hla['Instability.Index'].values[i], df_hla['Strand.Fraction'].values[i],
                               df_hla['Charge.at.pH.7.0'].values[i], df_hla['p_Aromaticity'].values[i],
                               df_hla['p_Strand.Fraction'].values[i], df_hla['p_Coil.Fraction'].values[i],
                               df_hla['p_Gravy'].values[i], df_hla['p_Amphipathicity'].values[i],
                               df_hla['p_GRAVY.Last.50'].values[i],
                               df_hla['p_Aliphatic.Index'].values[i], df_hla['p_Polar.Count'].values[i],
                               df_hla['p_N_Percent'].values[i], df_hla['p_C_Percent'].values[i],
                               df_hla['p_K_Percent'].values[i], df_hla['p_F_Percent'].values[i],
                               df_hla['p_P_Percent'].values[i], df_hla['p_S_Percent'].values[i],
                               df_hla['p_T_Percent'].values[i], df_hla['p_W_Percent'].values[i],
                               df_hla['p_V_Percent'].values[i], 0, protein_numerical(text_input)]
                    ext = joblib.load('extra_tree_hla_mhc1.pkl')
                    pred = ext.predict([hla_inp])[0]
                    ext_hla.append(pred)
                    print("The extra trees hla is ", pred)

                    lgbm = joblib.load('xgbr_hla_mhc1.pkl')
                    lgbm_hla.append(lgbm.predict([hla_inp])[0])
                    print("The LGBM prediction ", lgbm.predict([hla_inp])[0])

                    hist = joblib.load('hist_hla_mhc1.pkl')
                    hist_hla.append(hist.predict([hla_inp])[0])

                    print("The Hist_prediction ", hist.predict([hla_inp])[0])

                score_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                    "lgbm_prediction": lgbm_hla,
                    "extra_hla": ext_hla,
                    "hist_hla": hist_hla
                })

                score_df.to_csv("final_output.csv")
                score = pd.read_csv('final_output.csv')
                st.header("The File with score and values")
                st.write(score)

                print(score['Random_forest_Target'].values)
                print(score['Extra_tree_Target'].values)
                count_ones = score[['Extra_tree_Target', 'Random_forest_Target', 'bagging_Target']].sum(axis=1)
                score['Target'] = (count_ones > 2).astype(int)
                score.to_csv('target_final.csv')
                df_final = pd.read_csv('target_final.csv')
                df = df_final['extra_hla']
                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }

                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) * float(numeric_multiplier)
                    return total_score

                def find_nearest_hla(total_score, hla_strings, k=10):
                    nearest_hla_list = []
                    hla_strings_sorted = sorted(hla_strings,
                                                key=lambda hla: abs(calculate_numerical_score(hla) - total_score))
                    for hla in hla_strings_sorted[:k]:
                        nearest_hla_list.append(hla)
                    return nearest_hla_list

                data = pd.read_csv('hla.csv')
                hla_output = []
                pred = []
                hla_strings = [str(hla) for hla in data['HLA'].values]
                for i in range(len(df_final)):
                    nearest_hla_list = find_nearest_hla(df_final['extra_hla'].values[i], hla_strings, k=10)
                    print("10 Nearest HLA Strings:")
                    for i, hla in enumerate(nearest_hla_list, start=1):
                        print(hla)
                        pred.append(hla)
                        if len(pred) == 10:
                            hla_output.append(pred)
                            pred = []
                df_final['hla_values'] = hla_output
                df_final.to_csv("expected.csv")
                df_d3 = pd.read_csv("expected.csv")
                target = df_d3[['Extra_tree_Target', 'Random_forest_Target']].sum(axis=1)
                print('---------------------------------------------')
                print(target)
                df_d3['Target'] = (target > 1).astype(int)
                print(df_d3.Target)
                print('-------------------------------------------------------')
                df_tar = df_d3[df_d3['Target'] == 1]
                if len(df_tar.Target.values) <= 30:
                    df_tar.to_csv('target.csv')
                    df_tab = pd.read_csv('target.csv')
                    print(df_tab.columns)
                    col = ['start', 'end', 'Epitope', 'hla_values']
                    st.table(df_tab[col])
                else:
                    values = df_d3['XGB_predicted_score'].sort_values(ascending=False).values
                    val = []
                    for i in range(len(values)):
                        val.append(values[i])
                        if len(val) == 30:
                            break
                    epitope = []
                    hla = []
                    starts = []
                    ends = []
                    for i in val:
                        df_val = df_d3[df_d3['XGB_predicted_score'] == i]
                        epitope.append(df_val.Epitope)
                        hla.append(df_val.hla_values)
                        starts.append(df_val.start)
                        ends.append(df_val.end)
                    data_dict = {
                        'Epitope': epitope,
                        'HLA': hla,
                        'Start': starts,
                        'End': ends
                    }

                    df = pd.DataFrame(data_dict)
                    df = df.explode('Epitope').explode('HLA').explode('Start').explode('End')
                    df.reset_index(drop=True, inplace=True)
                    print(df)
                    st.write(df)
            elif prediction_option=='BOTH' and text_input:
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(6):
                    time.sleep(9)
                    progress_bar.progress((i + 1) * 100 // len(text2))
                    status_text.text(f'**⏳ Countdown: {((i + 1) * 100 // len(text2))}% ***')
                    st.write(f"[{i+1}] ",text2[i])
                status_text.text('# 💥 **Boom!** 💥')

                protein_sequence = text_input
                def find_epitopes(sequence, window_size=10):
                    epitopes = []
                    start = []
                    end = []
                    for i in range(len(sequence) - window_size + 1):
                        epitope = sequence[i:i + window_size]
                        epitopes.append(epitope)
                        start.append(i)
                        end.append(i + window_size - 1)
                    return (epitopes, start, end)

                def is_valid_protein_sequence(peptide_sequence):
                    valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(peptide_sequence) <= valid_letters

                def calculate_atom_counts(peptide_sequence):
                    atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }
                    for aa in peptide_sequence:
                        aa = aa.upper()
                        if aa in aa_info:
                            atom_counts['H'] += aa_info[aa][0]
                            atom_counts['C'] += aa_info[aa][1]
                            atom_counts['N'] += aa_info[aa][2]
                            atom_counts['O'] += aa_info[aa][3]
                            atom_counts['S'] += aa_info[aa][4]

                    return atom_counts

                def calculate_physicochemical_properties(peptide_sequence):
                    if not is_valid_protein_sequence(peptide_sequence):
                        return [None] * 35
                    protein_analyzer = ProteinAnalysis(peptide_sequence)
                    theoretical_pI = protein_analyzer.isoelectric_point()
                    aliphatic_index = sum(kd[aa] for aa in peptide_sequence) / len(peptide_sequence)
                    positive_residues = sum(peptide_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    negative_residues = sum(peptide_sequence.count(aa) for aa in ['D', 'E'])
                    aromatic_count = protein_analyzer.aromaticity() * len(peptide_sequence)
                    polar_amino_acids = set("STNQ")
                    non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    polar_count = sum(peptide_sequence.count(aa) for aa in polar_amino_acids)
                    nonpolar_count = sum(peptide_sequence.count(aa) for aa in non_polar_amino_acids)
                    amino_acid_composition = protein_analyzer.get_amino_acids_percent()
                    molecular_weight = protein_analyzer.molecular_weight()
                    instability_index = protein_analyzer.instability_index()
                    aromaticity = protein_analyzer.aromaticity()
                    helix_fraction = protein_analyzer.secondary_structure_fraction()[0]
                    strand_fraction = protein_analyzer.secondary_structure_fraction()[1]
                    coil_fraction = protein_analyzer.secondary_structure_fraction()[2]
                    charge_at_pH_7 = protein_analyzer.charge_at_pH(7.0)
                    gravy = protein_analyzer.gravy()
                    amphipathicity = calculate_amphipathicity(peptide_sequence)
                    gravy_last_50 = protein_analyzer.gravy()
                    molar_extinction_coefficient = protein_analyzer.molar_extinction_coefficient()

                    return [theoretical_pI, aliphatic_index, positive_residues, negative_residues, aromatic_count,
                            polar_count, nonpolar_count, amino_acid_composition, molecular_weight, instability_index,
                            aromaticity, helix_fraction, strand_fraction, coil_fraction, charge_at_pH_7, gravy,
                            amphipathicity,
                            gravy_last_50, molar_extinction_coefficient]

                def calculate_amphipathicity(peptide_sequence):
                    hydrophobic_moment_scale = kd
                    hydrophobic_moment = sum(hydrophobic_moment_scale[aa] for aa in peptide_sequence)
                    mean_hydrophobicity = hydrophobic_moment / len(peptide_sequence)
                    return hydrophobic_moment - mean_hydrophobicity

                def process_single_protein(peptide_sequence, start, end):
                    atom_counts = calculate_atom_counts(peptide_sequence)
                    physicochemical_properties = calculate_physicochemical_properties(peptide_sequence)
                    total_atoms = sum(atom_counts.values())

                    result_dict = {'epitope': peptide_sequence,
                                   'start': start,
                                   'end': end,
                                   'H_Count': atom_counts['H'],
                                   'C_Count': atom_counts['C'],
                                   'N_Count': atom_counts['N'],
                                   'O_Count': atom_counts['O'],
                                   'S_Count': atom_counts['S'],
                                   'TotalAtoms_Count': total_atoms}

                    result_dict.update({
                        'Theoretical.pI': physicochemical_properties[0],
                        'Aliphatic.Index': physicochemical_properties[1],
                        'Positive.Residues': physicochemical_properties[2],
                        'Negative.Residues': physicochemical_properties[3],
                        'Aromatic.Count': physicochemical_properties[4],
                        'Polar.Count': physicochemical_properties[5],
                        'Nonpolar.Count': physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W',
                                   'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        result_dict[aa + '_Percent'] = physicochemical_properties[7].get(aa, 0)

                    result_dict.update({
                        'Molecular.Weight': physicochemical_properties[8],
                        'Instability.Index': physicochemical_properties[9],
                        'Aromaticity': physicochemical_properties[10],
                        'Helix.Fraction': physicochemical_properties[11],
                        'Strand.Fraction': physicochemical_properties[12],
                        'Coil.Fraction': physicochemical_properties[13],
                        'Charge.at.pH.7.0': physicochemical_properties[14],
                        'Gravy': physicochemical_properties[15],
                        'Amphipathicity': physicochemical_properties[16],
                        'GRAVY.Last.50': physicochemical_properties[17],
                        'Molar.Extinction.Coefficient': physicochemical_properties[18]
                    })

                    return result_dict

                def p_is_valid_protein_sequence(protein_sequence):
                    p_valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(protein_sequence) <= p_valid_letters

                def p_calculate_atom_counts(protein_sequence):
                    p_atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    p_aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }

                    for aa in protein_sequence:
                        aa = aa.upper()
                        if aa in p_aa_info:
                            p_atom_counts['H'] += p_aa_info[aa][0]
                            p_atom_counts['C'] += p_aa_info[aa][1]
                            p_atom_counts['N'] += p_aa_info[aa][2]
                            p_atom_counts['O'] += p_aa_info[aa][3]
                            p_atom_counts['S'] += p_aa_info[aa][4]

                    return p_atom_counts

                def p_calculate_physicochemical_properties(protein_sequence):
                    if not p_is_valid_protein_sequence(protein_sequence):
                        return [None] * 35

                    p_protein_analyzer = ProteinAnalysis(protein_sequence)

                    p_theoretical_pI = p_protein_analyzer.isoelectric_point()
                    p_aliphatic_index = sum(kd[aa] for aa in protein_sequence) / len(protein_sequence)
                    p_positive_residues = sum(protein_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    p_negative_residues = sum(protein_sequence.count(aa) for aa in ['D', 'E'])
                    p_aromatic_count = p_protein_analyzer.aromaticity() * len(protein_sequence)
                    p_polar_amino_acids = set("STNQ")
                    p_non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    p_polar_count = sum(protein_sequence.count(aa) for aa in p_polar_amino_acids)
                    p_nonpolar_count = sum(protein_sequence.count(aa) for aa in p_non_polar_amino_acids)
                    p_amino_acid_composition = p_protein_analyzer.get_amino_acids_percent()
                    p_molecular_weight = p_protein_analyzer.molecular_weight()
                    p_instability_index = p_protein_analyzer.instability_index()
                    p_aromaticity = p_protein_analyzer.aromaticity()
                    p_helix_fraction = p_protein_analyzer.secondary_structure_fraction()[0]
                    p_strand_fraction = p_protein_analyzer.secondary_structure_fraction()[1]
                    p_coil_fraction = p_protein_analyzer.secondary_structure_fraction()[2]
                    p_charge_at_pH_7 = p_protein_analyzer.charge_at_pH(7.0)
                    p_gravy = p_protein_analyzer.gravy()
                    p_amphipathicity = p_calculate_amphipathicity(protein_sequence)
                    p_gravy_last_50 = p_protein_analyzer.gravy()
                    p_molar_extinction_coefficient = p_protein_analyzer.molar_extinction_coefficient()

                    return [p_theoretical_pI, p_aliphatic_index, p_positive_residues, p_negative_residues,
                            p_aromatic_count,
                            p_polar_count, p_nonpolar_count, p_amino_acid_composition, p_molecular_weight,
                            p_instability_index,
                            p_aromaticity, p_helix_fraction, p_strand_fraction, p_coil_fraction, p_charge_at_pH_7,
                            p_gravy,
                            p_amphipathicity,
                            p_gravy_last_50, p_molar_extinction_coefficient]

                def p_calculate_amphipathicity(protein_sequence):
                    p_hydrophobic_moment_scale = kd
                    p_hydrophobic_moment = sum(p_hydrophobic_moment_scale[aa] for aa in protein_sequence)
                    p_mean_hydrophobicity = p_hydrophobic_moment / len(protein_sequence)
                    return p_hydrophobic_moment - p_mean_hydrophobicity

                def p_process_single_protein(protein_sequence):
                    p_atom_counts = p_calculate_atom_counts(protein_sequence)
                    p_physicochemical_properties = p_calculate_physicochemical_properties(protein_sequence)
                    p_total_atoms = sum(p_atom_counts.values())

                    p_result_dict = {'p_Sequence': protein_sequence,
                                     'p_H_Count': p_atom_counts['H'],
                                     'p_C_Count': p_atom_counts['C'],
                                     'p_N_Count': p_atom_counts['N'],
                                     'p_O_Count': p_atom_counts['O'],
                                     'p_S_Count': p_atom_counts['S'],
                                     'p_TotalAtoms_Count': p_total_atoms}

                    p_result_dict.update({
                        'p_Theoretical.pI': p_physicochemical_properties[0],
                        'p_Aliphatic.Index': p_physicochemical_properties[1],
                        'p_Positive.Residues': p_physicochemical_properties[2],
                        'p_Negative.Residues': p_physicochemical_properties[3],
                        'p_Aromatic.Count': p_physicochemical_properties[4],
                        'p_Polar.Count': p_physicochemical_properties[5],
                        'p_Nonpolar.Count': p_physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W', 'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        p_result_dict['p_' + aa + '_Percent'] = p_physicochemical_properties[7].get(aa, 0)

                    p_result_dict.update({
                        'p_Molecular.Weight': p_physicochemical_properties[8],
                        'p_Instability.Index': p_physicochemical_properties[9],
                        'p_Aromaticity': p_physicochemical_properties[10],
                        'p_Helix.Fraction': p_physicochemical_properties[11],
                        'p_Strand.Fraction': p_physicochemical_properties[12],
                        'p_Coil.Fraction': p_physicochemical_properties[13],
                        'p_Charge.at.pH.7.0': p_physicochemical_properties[14],
                        'p_Gravy': p_physicochemical_properties[15],
                        'p_Amphipathicity': p_physicochemical_properties[16],
                        'p_GRAVY.Last.50': p_physicochemical_properties[17],
                        'p_Molar.Extinction.Coefficient': p_physicochemical_properties[18]
                    })

                    return p_result_dict

                r_result = p_process_single_protein(protein_sequence)
                epitopes = find_epitopes(protein_sequence, window_size=10)
                epi = []
                for i in range(len(epitopes[0])):
                    result = process_single_protein(epitopes[0][i], epitopes[1][i], epitopes[2][i])
                    epi.append(result)

                df = pd.DataFrame(epi)
                file_name = 'epitopes_results.csv'
                df.to_csv(file_name)
                df_d = pd.read_csv(file_name)
                st.header("The epitope information")
                st.write(df_d)

                pro = []
                for i in range(len(epi)):
                    r_result = p_process_single_protein(protein_sequence)
                    pro.append(r_result)

                df_p = pd.DataFrame(pro)
                file_name = 'p_Sequence.csv'
                df_p.to_csv(file_name)
                df_d1 = pd.read_csv(file_name)
                st.header("The Protein sequence information")
                st.write(df_d1)

                df1 = pd.read_csv('epitopes_results.csv')
                df2 = pd.read_csv('p_Sequence.csv')
                merged_df = pd.merge(df1, df2, how='inner')
                merged_df.to_csv('result.csv', index=False)
                print("Merged CSV file has been created.")

                final_res = pd.read_csv('result.csv')
                st.header('The CSV with epitope information')
                st.write(final_res)

                inps = ['start', 'end', 'R_Percent', 'D_Percent', 'Q_Percent', 'H_Percent',
                        'I_Percent', 'L_Percent',
                        'K_Percent', 'S_Percent', 'Theoretical.pI',
                        'Aliphatic.Index', 'Helix.Fraction', 'Charge.at.pH.7.0', 'Amphipathicity',
                        'p_Molecular.Weight', 'p_Instability.Index', 'p_Helix.Fraction',
                        'p_Amphipathicity', 'p_Aliphatic.Index',
                        'p_H_Count', 'p_C_Count', 'p_N_Count',
                        'p_O_Count', 'p_S_Count', 'p_TotalAtoms_Count',
                        'p_A_Percent',
                        'p_D_Percent',
                        'p_E_Percent', 'p_G_Percent',
                        'p_I_Percent', 'p_K_Percent',
                        'p_F_Percent', 'p_T_Percent',
                        'p_V_Percent',
                        ]
                columns_to_extract = [final_res[j].values[:len(final_res)] for j in inps]
                columns_data = dict(zip(inps, columns_to_extract))
                columns_df = pd.DataFrame(columns_data)
                columns_df.to_csv('extracted_columns.csv')
                bagging_pred = []
                extra_trees_pred = []
                random_forest_pred = []
                df = pd.read_csv('extracted_columns.csv')
                print(df.columns)
                st.header("The extracted Columns")
                st.write(df)
                for i in range(len(df)):
                    print(df.end.values[i])
                    print(
                        '-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print(f'FOR THE {final_res.epitope[i]} the value if 1-> epitope and o-> non-epitope')
                    print(
                        '------------------------------------------------------------------------------------------------------------')

                    inp = [df.start.values[i], df.end.values[i], df.R_Percent.values[i], df.D_Percent.values[i],
                           df.Q_Percent.values[i], df.H_Percent.values[i],
                           df.I_Percent.values[i], df.L_Percent.values[i], df.K_Percent.values[i],
                           df.S_Percent.values[i], df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                           df['Helix.Fraction'].values[i], df['Charge.at.pH.7.0'].values[i],
                           df['Amphipathicity'].values[i],
                           df['p_Molecular.Weight'].values[i], df['p_Instability.Index'].values[i],
                           df['p_Helix.Fraction'].values[i],
                           df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                           df['p_H_Count'].values[i],
                           df['p_C_Count'].values[i], df['p_N_Count'].values[i], df['p_O_Count'].values[i],
                           df['p_S_Count'].values[i], df['p_TotalAtoms_Count'].values[i], df['p_A_Percent'].values[i],
                           df['p_D_Percent'].values[i], df['p_E_Percent'].values[i], df['p_G_Percent'].values[i],
                           df['p_I_Percent'].values[i], df['p_K_Percent'].values[i], df['p_F_Percent'].values[i],
                           df['p_T_Percent'].values[i], df['p_V_Percent'].values[i]]

                    bagging = joblib.load('Bagging_tar_mhc1.pkl')
                    pred_bag = bagging.predict([inp])
                    bagging_pred.append(pred_bag[0])
                    print("The prediction using Bagging ", pred_bag)

                    extratree = joblib.load('extratree_tar_mhc1.pkl')
                    predict = extratree.predict([inp])
                    extra_trees_pred.append(predict[0])
                    print("The extra tree prediction ", predict)

                    randomfor = joblib.load('randomforest_tar_mhc1.pkl')
                    random_pred = randomfor.predict([inp])
                    random_forest_pred.append(random_pred[0])
                    print("The random forest ", random_pred)

                df = pd.read_csv('result.csv')
                print(df.columns)
                epitopes = []
                xg_boost = []
                lgbm_score = []
                start_val = []
                end_val = []

                for i in range(len(df)):
                    print('-------------------------------------------')
                    epitopes.append(df.epitope.values[i])
                    start_val.append(df.start.values[i])
                    end_val.append(df.end.values[i])
                    print("---------------------------------------------")
                    print(df.epitope.values[i])

                    print('---------------------------------------------')
                    score_inp = [df.start.values[i], df.end.values[i],
                                 df['A_Percent'].values[i], df['R_Percent'].values[i],
                                 df['N_Percent'].values[i], df['D_Percent'].values[i],
                                 df['C_Percent'].values[i], df['E_Percent'].values[i],
                                 df['Q_Percent'].values[i], df['G_Percent'].values[i], df['H_Percent'].values[i],
                                 df['I_Percent'].values[i], df['L_Percent'].values[i], df['K_Percent'].values[i],
                                 df['M_Percent'].values[i], df['F_Percent'].values[i], df['P_Percent'].values[i],
                                 df['S_Percent'].values[i], df['T_Percent'].values[i], df['W_Percent'].values[i],
                                 df['Y_Percent'].values[i], df['V_Percent'].values[i],
                                 df['H_Count'].values[i], df['C_Count'].values[i], df['N_Count'].values[i],
                                 df['S_Count'].values[i],
                                 df['TotalAtoms_Count'].values[i],
                                 df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                                 df['Positive.Residues'].values[i], df['Negative.Residues'].values[i],
                                 df['Aromatic.Count'].values[i], df['Polar.Count'].values[i],
                                 df['Nonpolar.Count'].values[i], df['Molecular.Weight'].values[i],
                                 df['Instability.Index'].values[i], df['Aromaticity'].values[i],
                                 df['Helix.Fraction'].values[i], df['Strand.Fraction'].values[i],
                                 df['Coil.Fraction'].values[i],
                                 df['Charge.at.pH.7.0'].values[i], df['Amphipathicity'].values[i],
                                 df['GRAVY.Last.50'].values[i],
                                 df['p_Instability.Index'].values[i], df['p_Helix.Fraction'].values[i],
                                 df['p_Strand.Fraction'].values[i], df['p_Coil.Fraction'].values[i],
                                 df['p_Charge.at.pH.7.0'].values[i],
                                 df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                                 df['p_Aromatic.Count'].values[i], df['p_Nonpolar.Count'].values[i],
                                 df['p_H_Count'].values[i],
                                 df['p_C_Count'].values[i], df['p_O_Count'].values[i],
                                 df['p_TotalAtoms_Count'].values[i],
                                 df['p_R_Percent'].values[i],
                                 df['p_N_Percent'].values[i], df['p_D_Percent'].values[i],
                                 df['p_E_Percent'].values[i],
                                 df['p_L_Percent'].values[i],
                                 df['p_T_Percent'].values[i], df['p_W_Percent'].values[i]]

                    xgb = joblib.load('xgb_score_mhc1.pkl')
                    xgb_pred = xgb.predict([score_inp])
                    print("The xgb_pred ", xgb_pred)
                    xg_boost.append(xgb_pred[0])

                    lgb = joblib.load('lgb_score_mhc1.pkl')
                    lgbm_prediction = lgb.predict([score_inp])
                    lgbm_score.append(lgbm_prediction[0])
                    print('The lgbm prediction ', lgbm_prediction)

                kolaskar_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                })

                kolaskar_df.to_csv('kolaskar.csv')
                df_kolaskar = pd.read_csv("kolaskar.csv")
                st.header('The Kolaskar score information')
                st.write(df_kolaskar)

                def prot_to_num(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }

                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) + float(numeric_multiplier)
                    return total_score

                def protein_numerical(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                protein_sequence = "AAAALNGVDRRSLQRSARLALEVLERAKRRAVDWHALERPKGCMGVLAREAPHLEKQPAAGPQRVLPGEKYYSSVPEEGGATHVYRYHRGESKLHMCLDIGNGQAENISKDLYIEVYPGTYSVTVGSNDLTKKTHVVAVDSGQSVDLVFPV"
                df_hla = pd.read_csv('result.csv')
                ext_hla = []
                lgbm_hla = []
                hist_hla = []
                for i in range(len(df_hla)):
                    print(f"The HLA Prediction for {df_hla.epitope.values[i]}")
                    hla_inp = [extra_trees_pred[i], df_hla['C_Percent'].values[i], df_hla['Q_Percent'].values[i],
                               df_hla['G_Percent'].values[i],
                               df_hla['K_Percent'].values[i], df_hla['P_Percent'].values[i],
                               df_hla['S_Percent'].values[i],
                               df_hla['T_Percent'].values[i], df_hla['W_Percent'].values[i],
                               df_hla['H_Count'].values[i],
                               df_hla['C_Count'].values[i],
                               df_hla['N_Count'].values[i], df_hla['O_Count'].values[i],
                               df_hla['TotalAtoms_Count'].values[i],
                               df_hla['Theoretical.pI'].values[i],
                               df_hla['Positive.Residues'].values[i], df_hla['Negative.Residues'].values[i],
                               df_hla['Aromatic.Count'].values[i], df_hla['Polar.Count'].values[i],
                               df_hla['Nonpolar.Count'].values[i], df_hla['Molecular.Weight'].values[i],
                               df_hla['Instability.Index'].values[i], df_hla['Strand.Fraction'].values[i],
                               df_hla['Charge.at.pH.7.0'].values[i], df_hla['p_Aromaticity'].values[i],
                               df_hla['p_Strand.Fraction'].values[i], df_hla['p_Coil.Fraction'].values[i],
                               df_hla['p_Gravy'].values[i], df_hla['p_Amphipathicity'].values[i],
                               df_hla['p_GRAVY.Last.50'].values[i],
                               df_hla['p_Aliphatic.Index'].values[i], df_hla['p_Polar.Count'].values[i],
                               df_hla['p_N_Percent'].values[i], df_hla['p_C_Percent'].values[i],
                               df_hla['p_K_Percent'].values[i], df_hla['p_F_Percent'].values[i],
                               df_hla['p_P_Percent'].values[i], df_hla['p_S_Percent'].values[i],
                               df_hla['p_T_Percent'].values[i], df_hla['p_W_Percent'].values[i],
                               df_hla['p_V_Percent'].values[i], 0, protein_numerical(text_input)]
                    ext = joblib.load('extra_tree_hla_mhc1.pkl')
                    pred = ext.predict([hla_inp])[0]
                    ext_hla.append(pred)
                    print("The extra trees hla is ", pred)

                    lgbm = joblib.load('xgbr_hla_mhc1.pkl')
                    lgbm_hla.append(lgbm.predict([hla_inp])[0])
                    print("The LGBM prediction ", lgbm.predict([hla_inp])[0])

                    hist = joblib.load('hist_hla_mhc1.pkl')
                    hist_hla.append(hist.predict([hla_inp])[0])

                    print("The Hist_prediction ", hist.predict([hla_inp])[0])

                score_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                    "lgbm_prediction": lgbm_hla,
                    "extra_hla": ext_hla,
                    "hist_hla": hist_hla
                })

                score_df.to_csv("final_output.csv")
                score = pd.read_csv('final_output.csv')
                st.header("The File with score and values")
                st.write(score)

                print(score['Random_forest_Target'].values)
                print(score['Extra_tree_Target'].values)
                count_ones = score[['Extra_tree_Target', 'Random_forest_Target', 'bagging_Target']].sum(axis=1)
                score['Target'] = (count_ones > 2).astype(int)
                score.to_csv('target_final.csv')
                df_final = pd.read_csv('target_final.csv')
                df = df_final['extra_hla']
                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }

                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) * float(numeric_multiplier)
                    return total_score

                def find_nearest_hla(total_score, hla_strings, k=10):
                    nearest_hla_list = []
                    hla_strings_sorted = sorted(hla_strings,
                                                key=lambda hla: abs(calculate_numerical_score(hla) - total_score))
                    for hla in hla_strings_sorted[:k]:
                        nearest_hla_list.append(hla)
                    return nearest_hla_list

                data = pd.read_csv('hla.csv')
                hla_output = []
                pred = []
                hla_strings = [str(hla) for hla in data['HLA'].values]
                for i in range(len(df_final)):
                    nearest_hla_list = find_nearest_hla(df_final['extra_hla'].values[i], hla_strings, k=10)
                    print("10 Nearest HLA Strings:")
                    for i, hla in enumerate(nearest_hla_list, start=1):
                        print(hla)
                        pred.append(hla)
                        if len(pred) == 10:
                            hla_output.append(pred)
                            pred = []
                df_final['hla_values'] = hla_output
                df_final.to_csv("expected.csv")
                df_d3 = pd.read_csv("expected.csv")
                target = df_d3[['Extra_tree_Target', 'Random_forest_Target']].sum(axis=1)
                print('---------------------------------------------')
                print(target)
                df_d3['Target'] = (target > 1).astype(int)
                print(df_d3.Target)
                print('-------------------------------------------------------')
                df_tar = df_d3[df_d3['Target'] == 1]
                if len(df_tar.Target.values) <= 30:
                    df_tar.to_csv('target.csv')
                    df_tab = pd.read_csv('target.csv')
                    print(df_tab.columns)
                    col = ['start', 'end', 'Epitope', 'hla_values']
                    st.table(df_tab[col])
                else:
                    values = df_d3['XGB_predicted_score'].sort_values(ascending=False).values
                    val = []
                    for i in range(len(values)):
                        val.append(values[i])
                        if len(val) == 30:
                            break
                    epitope = []
                    hla = []
                    starts = []
                    ends = []
                    for i in val:
                        df_val = df_d3[df_d3['XGB_predicted_score'] == i]
                        epitope.append(df_val.Epitope)
                        hla.append(df_val.hla_values)
                        starts.append(df_val.start)
                        ends.append(df_val.end)
                    data_dict = {
                        'Epitope': epitope,
                        'HLA': hla,
                        'Start': starts,
                        'End': ends
                    }
                    df = pd.DataFrame(data_dict)
                    df = df.explode('Epitope').explode('HLA').explode('Start').explode('End')
                    df.reset_index(drop=True, inplace=True)
                    print(df)
                    st.write(df)

                protein_sequence = text_input

                def find_epitopes(sequence, window_size=15):
                    epitopes = []
                    start = []
                    end = []
                    for i in range(len(sequence) - window_size + 1):
                        epitope = sequence[i:i + window_size]
                        epitopes.append(epitope)
                        start.append(i)
                        end.append(i + window_size - 1)
                    return (epitopes, start, end)

                def is_valid_protein_sequence(peptide_sequence):
                    valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(peptide_sequence) <= valid_letters

                def calculate_atom_counts(peptide_sequence):
                    atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }
                    for aa in peptide_sequence:
                        aa = aa.upper()
                        if aa in aa_info:
                            atom_counts['H'] += aa_info[aa][0]
                            atom_counts['C'] += aa_info[aa][1]
                            atom_counts['N'] += aa_info[aa][2]
                            atom_counts['O'] += aa_info[aa][3]
                            atom_counts['S'] += aa_info[aa][4]

                    return atom_counts

                def calculate_physicochemical_properties(peptide_sequence):
                    if not is_valid_protein_sequence(peptide_sequence):
                        return [None] * 35
                    protein_analyzer = ProteinAnalysis(peptide_sequence)
                    theoretical_pI = protein_analyzer.isoelectric_point()
                    aliphatic_index = sum(kd[aa] for aa in peptide_sequence) / len(peptide_sequence)
                    positive_residues = sum(peptide_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    negative_residues = sum(peptide_sequence.count(aa) for aa in ['D', 'E'])
                    aromatic_count = protein_analyzer.aromaticity() * len(peptide_sequence)
                    polar_amino_acids = set("STNQ")
                    non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    polar_count = sum(peptide_sequence.count(aa) for aa in polar_amino_acids)
                    nonpolar_count = sum(peptide_sequence.count(aa) for aa in non_polar_amino_acids)
                    amino_acid_composition = protein_analyzer.get_amino_acids_percent()
                    molecular_weight = protein_analyzer.molecular_weight()
                    instability_index = protein_analyzer.instability_index()
                    aromaticity = protein_analyzer.aromaticity()
                    helix_fraction = protein_analyzer.secondary_structure_fraction()[0]
                    strand_fraction = protein_analyzer.secondary_structure_fraction()[1]
                    coil_fraction = protein_analyzer.secondary_structure_fraction()[2]
                    charge_at_pH_7 = protein_analyzer.charge_at_pH(7.0)
                    gravy = protein_analyzer.gravy()
                    amphipathicity = calculate_amphipathicity(peptide_sequence)
                    gravy_last_50 = protein_analyzer.gravy()
                    molar_extinction_coefficient = protein_analyzer.molar_extinction_coefficient()

                    return [theoretical_pI, aliphatic_index, positive_residues, negative_residues, aromatic_count,
                            polar_count, nonpolar_count, amino_acid_composition, molecular_weight, instability_index,
                            aromaticity, helix_fraction, strand_fraction, coil_fraction, charge_at_pH_7, gravy,
                            amphipathicity,
                            gravy_last_50, molar_extinction_coefficient]

                def calculate_amphipathicity(peptide_sequence):
                    hydrophobic_moment_scale = kd
                    hydrophobic_moment = sum(hydrophobic_moment_scale[aa] for aa in peptide_sequence)
                    mean_hydrophobicity = hydrophobic_moment / len(peptide_sequence)
                    return hydrophobic_moment - mean_hydrophobicity

                def process_single_protein(peptide_sequence, start, end):
                    atom_counts = calculate_atom_counts(peptide_sequence)
                    physicochemical_properties = calculate_physicochemical_properties(peptide_sequence)
                    total_atoms = sum(atom_counts.values())

                    result_dict = {'epitope': peptide_sequence,
                                   'start': start,
                                   'end': end,
                                   'H_Count': atom_counts['H'],
                                   'C_Count': atom_counts['C'],
                                   'N_Count': atom_counts['N'],
                                   'O_Count': atom_counts['O'],
                                   'S_Count': atom_counts['S'],
                                   'TotalAtoms_Count': total_atoms}

                    result_dict.update({
                        'Theoretical.pI': physicochemical_properties[0],
                        'Aliphatic.Index': physicochemical_properties[1],
                        'Positive.Residues': physicochemical_properties[2],
                        'Negative.Residues': physicochemical_properties[3],
                        'Aromatic.Count': physicochemical_properties[4],
                        'Polar.Count': physicochemical_properties[5],
                        'Nonpolar.Count': physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W',
                                   'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        result_dict[aa + '_Percent'] = physicochemical_properties[7].get(aa, 0)

                    result_dict.update({
                        'Molecular.Weight': physicochemical_properties[8],
                        'Instability.Index': physicochemical_properties[9],
                        'Aromaticity': physicochemical_properties[10],
                        'Helix.Fraction': physicochemical_properties[11],
                        'Strand.Fraction': physicochemical_properties[12],
                        'Coil.Fraction': physicochemical_properties[13],
                        'Charge.at.pH.7.0': physicochemical_properties[14],
                        'Gravy': physicochemical_properties[15],
                        'Amphipathicity': physicochemical_properties[16],
                        'GRAVY.Last.50': physicochemical_properties[17],
                        'Molar.Extinction.Coefficient': physicochemical_properties[18]
                    })

                    return result_dict

                def p_is_valid_protein_sequence(protein_sequence):
                    p_valid_letters = set("ACDEFGHIKLMNPQRSTVWY")
                    return set(protein_sequence) <= p_valid_letters

                def p_calculate_atom_counts(protein_sequence):
                    p_atom_counts = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'S': 0}
                    p_aa_info = {
                        'A': [5, 3, 1, 1, 0], 'R': [17, 6, 4, 2, 0], 'N': [8, 4, 2, 2, 0], 'D': [7, 4, 1, 3, 0],
                        'C': [7, 3, 1, 1, 1], 'E': [9, 5, 1, 3, 0], 'Q': [10, 5, 2, 2, 0], 'G': [3, 2, 1, 1, 0],
                        'H': [11, 6, 3, 2, 0], 'I': [11, 6, 1, 2, 0], 'L': [11, 6, 1, 2, 0], 'K': [14, 6, 2, 2, 0],
                        'M': [11, 5, 1, 2, 1], 'F': [11, 9, 1, 1, 0], 'P': [9, 5, 1, 1, 0], 'S': [9, 3, 1, 2, 0],
                        'T': [11, 4, 1, 2, 0], 'W': [14, 11, 2, 1, 0], 'Y': [12, 6, 1, 3, 0], 'V': [9, 5, 1, 1, 0]
                    }

                    for aa in protein_sequence:
                        aa = aa.upper()
                        if aa in p_aa_info:
                            p_atom_counts['H'] += p_aa_info[aa][0]
                            p_atom_counts['C'] += p_aa_info[aa][1]
                            p_atom_counts['N'] += p_aa_info[aa][2]
                            p_atom_counts['O'] += p_aa_info[aa][3]
                            p_atom_counts['S'] += p_aa_info[aa][4]

                    return p_atom_counts

                def p_calculate_physicochemical_properties(protein_sequence):
                    if not p_is_valid_protein_sequence(protein_sequence):
                        return [None] * 35

                    p_protein_analyzer = ProteinAnalysis(protein_sequence)

                    p_theoretical_pI = p_protein_analyzer.isoelectric_point()
                    p_aliphatic_index = sum(kd[aa] for aa in protein_sequence) / len(protein_sequence)
                    p_positive_residues = sum(protein_sequence.count(aa) for aa in ['R', 'K', 'H'])
                    p_negative_residues = sum(protein_sequence.count(aa) for aa in ['D', 'E'])
                    p_aromatic_count = p_protein_analyzer.aromaticity() * len(protein_sequence)
                    p_polar_amino_acids = set("STNQ")
                    p_non_polar_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
                    p_polar_count = sum(protein_sequence.count(aa) for aa in p_polar_amino_acids)
                    p_nonpolar_count = sum(protein_sequence.count(aa) for aa in p_non_polar_amino_acids)
                    p_amino_acid_composition = p_protein_analyzer.get_amino_acids_percent()
                    p_molecular_weight = p_protein_analyzer.molecular_weight()
                    p_instability_index = p_protein_analyzer.instability_index()
                    p_aromaticity = p_protein_analyzer.aromaticity()
                    p_helix_fraction = p_protein_analyzer.secondary_structure_fraction()[0]
                    p_strand_fraction = p_protein_analyzer.secondary_structure_fraction()[1]
                    p_coil_fraction = p_protein_analyzer.secondary_structure_fraction()[2]
                    p_charge_at_pH_7 = p_protein_analyzer.charge_at_pH(7.0)
                    p_gravy = p_protein_analyzer.gravy()
                    p_amphipathicity = p_calculate_amphipathicity(protein_sequence)
                    p_gravy_last_50 = p_protein_analyzer.gravy()
                    p_molar_extinction_coefficient = p_protein_analyzer.molar_extinction_coefficient()

                    return [p_theoretical_pI, p_aliphatic_index, p_positive_residues, p_negative_residues,
                            p_aromatic_count,
                            p_polar_count, p_nonpolar_count, p_amino_acid_composition, p_molecular_weight,
                            p_instability_index,
                            p_aromaticity, p_helix_fraction, p_strand_fraction, p_coil_fraction, p_charge_at_pH_7,
                            p_gravy,
                            p_amphipathicity,
                            p_gravy_last_50, p_molar_extinction_coefficient]

                def p_calculate_amphipathicity(protein_sequence):
                    p_hydrophobic_moment_scale = kd
                    p_hydrophobic_moment = sum(p_hydrophobic_moment_scale[aa] for aa in protein_sequence)
                    p_mean_hydrophobicity = p_hydrophobic_moment / len(protein_sequence)
                    return p_hydrophobic_moment - p_mean_hydrophobicity

                def p_process_single_protein(protein_sequence):
                    p_atom_counts = p_calculate_atom_counts(protein_sequence)
                    p_physicochemical_properties = p_calculate_physicochemical_properties(protein_sequence)
                    p_total_atoms = sum(p_atom_counts.values())

                    p_result_dict = {'p_Sequence': protein_sequence,
                                     'p_H_Count': p_atom_counts['H'],
                                     'p_C_Count': p_atom_counts['C'],
                                     'p_N_Count': p_atom_counts['N'],
                                     'p_O_Count': p_atom_counts['O'],
                                     'p_S_Count': p_atom_counts['S'],
                                     'p_TotalAtoms_Count': p_total_atoms}

                    p_result_dict.update({
                        'p_Theoretical.pI': p_physicochemical_properties[0],
                        'p_Aliphatic.Index': p_physicochemical_properties[1],
                        'p_Positive.Residues': p_physicochemical_properties[2],
                        'p_Negative.Residues': p_physicochemical_properties[3],
                        'p_Aromatic.Count': p_physicochemical_properties[4],
                        'p_Polar.Count': p_physicochemical_properties[5],
                        'p_Nonpolar.Count': p_physicochemical_properties[6]
                    })

                    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                                   'W', 'Y', 'V']
                    for i, aa in enumerate(amino_acids):
                        p_result_dict['p_' + aa + '_Percent'] = p_physicochemical_properties[7].get(aa, 0)

                    p_result_dict.update({
                        'p_Molecular.Weight': p_physicochemical_properties[8],
                        'p_Instability.Index': p_physicochemical_properties[9],
                        'p_Aromaticity': p_physicochemical_properties[10],
                        'p_Helix.Fraction': p_physicochemical_properties[11],
                        'p_Strand.Fraction': p_physicochemical_properties[12],
                        'p_Coil.Fraction': p_physicochemical_properties[13],
                        'p_Charge.at.pH.7.0': p_physicochemical_properties[14],
                        'p_Gravy': p_physicochemical_properties[15],
                        'p_Amphipathicity': p_physicochemical_properties[16],
                        'p_GRAVY.Last.50': p_physicochemical_properties[17],
                        'p_Molar.Extinction.Coefficient': p_physicochemical_properties[18]
                    })

                    return p_result_dict

                r_result = p_process_single_protein(protein_sequence)
                epitopes = find_epitopes(protein_sequence, window_size=15)
                epi = []
                for i in range(len(epitopes[0])):
                    result = process_single_protein(epitopes[0][i], epitopes[1][i], epitopes[2][i])
                    epi.append(result)

                df = pd.DataFrame(epi)
                file_name = 'epitopes_results.csv'
                df.to_csv(file_name)
                df_d = pd.read_csv(file_name)
                st.header("The epitope information")
                st.write(df_d)

                pro = []
                for i in range(len(epi)):
                    r_result = p_process_single_protein(protein_sequence)
                    pro.append(r_result)

                df_p = pd.DataFrame(pro)
                file_name = 'p_Sequence.csv'
                df_p.to_csv(file_name)
                df_d1 = pd.read_csv(file_name)
                st.header("The Protein sequence information")
                st.write(df_d1)

                df1 = pd.read_csv('epitopes_results.csv')
                df2 = pd.read_csv('p_Sequence.csv')
                merged_df = pd.merge(df1, df2, how='inner')
                merged_df.to_csv('result.csv', index=False)
                print("Merged CSV file has been created.")

                final_res = pd.read_csv('result.csv')
                st.header('The CSV with epitope information')
                st.write(final_res)

                inps = ['start', 'end', 'R_Percent', 'D_Percent', 'Q_Percent', 'H_Percent',
                        'I_Percent', 'L_Percent',
                        'K_Percent', 'S_Percent', 'Theoretical.pI',
                        'Aliphatic.Index', 'Helix.Fraction', 'Charge.at.pH.7.0', 'Amphipathicity',
                        'p_Molecular.Weight', 'p_Instability.Index', 'p_Helix.Fraction',
                        'p_Amphipathicity', 'p_Aliphatic.Index',
                        'p_H_Count', 'p_C_Count', 'p_N_Count',
                        'p_O_Count', 'p_S_Count', 'p_TotalAtoms_Count',
                        'p_A_Percent',
                        'p_D_Percent',
                        'p_E_Percent', 'p_G_Percent',
                        'p_I_Percent', 'p_K_Percent',
                        'p_F_Percent', 'p_T_Percent',
                        'p_V_Percent',
                        ]
                columns_to_extract = [final_res[j].values[:len(final_res)] for j in inps]
                columns_data = dict(zip(inps, columns_to_extract))
                columns_df = pd.DataFrame(columns_data)
                columns_df.to_csv('extracted_columns.csv')
                bagging_pred = []
                extra_trees_pred = []
                random_forest_pred = []
                df = pd.read_csv('extracted_columns.csv')
                print(df.columns)
                st.header("The extracted Columns")
                st.write(df)
                for i in range(len(df)):
                    print(df.end.values[i])
                    print(
                        '-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print(f'FOR THE {final_res.epitope[i]} the value if 1-> epitope and o-> non-epitope')
                    print(
                        '------------------------------------------------------------------------------------------------------------')

                    inp = [df.start.values[i], df.end.values[i], df.R_Percent.values[i], df.D_Percent.values[i],
                           df.Q_Percent.values[i], df.H_Percent.values[i],
                           df.I_Percent.values[i], df.L_Percent.values[i], df.K_Percent.values[i],
                           df.S_Percent.values[i], df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                           df['Helix.Fraction'].values[i], df['Charge.at.pH.7.0'].values[i],
                           df['Amphipathicity'].values[i],
                           df['p_Molecular.Weight'].values[i], df['p_Instability.Index'].values[i],
                           df['p_Helix.Fraction'].values[i],
                           df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                           df['p_H_Count'].values[i],
                           df['p_C_Count'].values[i], df['p_N_Count'].values[i], df['p_O_Count'].values[i],
                           df['p_S_Count'].values[i], df['p_TotalAtoms_Count'].values[i], df['p_A_Percent'].values[i],
                           df['p_D_Percent'].values[i], df['p_E_Percent'].values[i], df['p_G_Percent'].values[i],
                           df['p_I_Percent'].values[i], df['p_K_Percent'].values[i], df['p_F_Percent'].values[i],
                           df['p_T_Percent'].values[i], df['p_V_Percent'].values[i]]

                    bagging = joblib.load('Bagging_tar_mhc1.pkl')
                    pred_bag = bagging.predict([inp])
                    bagging_pred.append(pred_bag[0])
                    print("The prediction using Bagging ", pred_bag)

                    extratree = joblib.load('extratree_tar_mhc1.pkl')
                    predict = extratree.predict([inp])
                    extra_trees_pred.append(predict[0])
                    print("The extra tree prediction ", predict)

                    randomfor = joblib.load('randomforest_tar_mhc1.pkl')
                    random_pred = randomfor.predict([inp])
                    random_forest_pred.append(random_pred[0])
                    print("The random forest ", random_pred)

                df = pd.read_csv('result.csv')
                print(df.columns)
                epitopes = []
                xg_boost = []
                lgbm_score = []
                start_val = []
                end_val = []

                for i in range(len(df)):
                    print('-------------------------------------------')
                    epitopes.append(df.epitope.values[i])
                    start_val.append(df.start.values[i])
                    end_val.append(df.end.values[i])
                    print("---------------------------------------------")
                    print(df.epitope.values[i])

                    print('---------------------------------------------')
                    score_inp = [df.start.values[i], df.end.values[i],
                                 df['A_Percent'].values[i], df['R_Percent'].values[i],
                                 df['N_Percent'].values[i], df['D_Percent'].values[i],
                                 df['C_Percent'].values[i], df['E_Percent'].values[i],
                                 df['Q_Percent'].values[i], df['G_Percent'].values[i], df['H_Percent'].values[i],
                                 df['I_Percent'].values[i], df['L_Percent'].values[i], df['K_Percent'].values[i],
                                 df['M_Percent'].values[i], df['F_Percent'].values[i], df['P_Percent'].values[i],
                                 df['S_Percent'].values[i], df['T_Percent'].values[i], df['W_Percent'].values[i],
                                 df['Y_Percent'].values[i], df['V_Percent'].values[i],
                                 df['H_Count'].values[i], df['C_Count'].values[i], df['N_Count'].values[i],
                                 df['S_Count'].values[i],
                                 df['TotalAtoms_Count'].values[i],
                                 df['Theoretical.pI'].values[i], df['Aliphatic.Index'].values[i],
                                 df['Positive.Residues'].values[i], df['Negative.Residues'].values[i],
                                 df['Aromatic.Count'].values[i], df['Polar.Count'].values[i],
                                 df['Nonpolar.Count'].values[i], df['Molecular.Weight'].values[i],
                                 df['Instability.Index'].values[i], df['Aromaticity'].values[i],
                                 df['Helix.Fraction'].values[i], df['Strand.Fraction'].values[i],
                                 df['Coil.Fraction'].values[i],
                                 df['Charge.at.pH.7.0'].values[i], df['Amphipathicity'].values[i],
                                 df['GRAVY.Last.50'].values[i],
                                 df['p_Instability.Index'].values[i], df['p_Helix.Fraction'].values[i],
                                 df['p_Strand.Fraction'].values[i], df['p_Coil.Fraction'].values[i],
                                 df['p_Charge.at.pH.7.0'].values[i],
                                 df['p_Amphipathicity'].values[i], df['p_Aliphatic.Index'].values[i],
                                 df['p_Aromatic.Count'].values[i], df['p_Nonpolar.Count'].values[i],
                                 df['p_H_Count'].values[i],
                                 df['p_C_Count'].values[i], df['p_O_Count'].values[i],
                                 df['p_TotalAtoms_Count'].values[i],
                                 df['p_R_Percent'].values[i],
                                 df['p_N_Percent'].values[i], df['p_D_Percent'].values[i],
                                 df['p_E_Percent'].values[i],
                                 df['p_L_Percent'].values[i],
                                 df['p_T_Percent'].values[i], df['p_W_Percent'].values[i]]

                    xgb = joblib.load('xgb_score_mhc1.pkl')
                    xgb_pred = xgb.predict([score_inp])
                    print("The xgb_pred ", xgb_pred)
                    xg_boost.append(xgb_pred[0])

                    lgb = joblib.load('lgb_score_mhc1.pkl')
                    lgbm_prediction = lgb.predict([score_inp])
                    lgbm_score.append(lgbm_prediction[0])
                    print('The lgbm prediction ', lgbm_prediction)

                kolaskar_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                })

                kolaskar_df.to_csv('kolaskar.csv')
                df_kolaskar = pd.read_csv("kolaskar.csv")
                st.header('The Kolaskar score information')
                st.write(df_kolaskar)

                def prot_to_num(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }

                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) + float(numeric_multiplier)
                    return total_score

                def protein_numerical(sequence):
                    if isinstance(sequence, str):
                        aa_hydrophobicity = {
                            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                        }
                        numerical_seq = [aa_hydrophobicity.get(aa, 0.5) for aa in
                                         sequence.upper()]  # Convert to uppercase
                        return sum(numerical_seq) / len(numerical_seq) if len(
                            numerical_seq) > 0 else 0.5  # Handle empty sequences
                    else:
                        raise TypeError("Input must be a string representing a protein sequence.")

                protein_sequence = "AAAALNGVDRRSLQRSARLALEVLERAKRRAVDWHALERPKGCMGVLAREAPHLEKQPAAGPQRVLPGEKYYSSVPEEGGATHVYRYHRGESKLHMCLDIGNGQAENISKDLYIEVYPGTYSVTVGSNDLTKKTHVVAVDSGQSVDLVFPV"
                df_hla = pd.read_csv('result.csv')
                ext_hla = []
                lgbm_hla = []
                hist_hla = []
                for i in range(len(df_hla)):
                    print(f"The HLA Prediction for {df_hla.epitope.values[i]}")
                    hla_inp = [extra_trees_pred[i], df_hla['C_Percent'].values[i], df_hla['Q_Percent'].values[i],
                               df_hla['G_Percent'].values[i],
                               df_hla['K_Percent'].values[i], df_hla['P_Percent'].values[i],
                               df_hla['S_Percent'].values[i],
                               df_hla['T_Percent'].values[i], df_hla['W_Percent'].values[i],
                               df_hla['H_Count'].values[i],
                               df_hla['C_Count'].values[i],
                               df_hla['N_Count'].values[i], df_hla['O_Count'].values[i],
                               df_hla['TotalAtoms_Count'].values[i],
                               df_hla['Theoretical.pI'].values[i],
                               df_hla['Positive.Residues'].values[i], df_hla['Negative.Residues'].values[i],
                               df_hla['Aromatic.Count'].values[i], df_hla['Polar.Count'].values[i],
                               df_hla['Nonpolar.Count'].values[i], df_hla['Molecular.Weight'].values[i],
                               df_hla['Instability.Index'].values[i], df_hla['Strand.Fraction'].values[i],
                               df_hla['Charge.at.pH.7.0'].values[i], df_hla['p_Aromaticity'].values[i],
                               df_hla['p_Strand.Fraction'].values[i], df_hla['p_Coil.Fraction'].values[i],
                               df_hla['p_Gravy'].values[i], df_hla['p_Amphipathicity'].values[i],
                               df_hla['p_GRAVY.Last.50'].values[i],
                               df_hla['p_Aliphatic.Index'].values[i], df_hla['p_Polar.Count'].values[i],
                               df_hla['p_N_Percent'].values[i], df_hla['p_C_Percent'].values[i],
                               df_hla['p_K_Percent'].values[i], df_hla['p_F_Percent'].values[i],
                               df_hla['p_P_Percent'].values[i], df_hla['p_S_Percent'].values[i],
                               df_hla['p_T_Percent'].values[i], df_hla['p_W_Percent'].values[i],
                               df_hla['p_V_Percent'].values[i], 0, protein_numerical(text_input)]
                    ext = joblib.load('extra_tree_hla_mhc1.pkl')
                    pred = ext.predict([hla_inp])[0]
                    ext_hla.append(pred)
                    print("The extra trees hla is ", pred)

                    lgbm = joblib.load('xgbr_hla_mhc1.pkl')
                    lgbm_hla.append(lgbm.predict([hla_inp])[0])
                    print("The LGBM prediction ", lgbm.predict([hla_inp])[0])

                    hist = joblib.load('hist_hla_mhc1.pkl')
                    hist_hla.append(hist.predict([hla_inp])[0])

                    print("The Hist_prediction ", hist.predict([hla_inp])[0])

                score_df = pd.DataFrame({
                    "start": start_val,
                    "end": end_val,
                    "Epitope": epitopes,
                    "XGB_predicted_score": xg_boost,
                    "light_gbm_predicted_score": lgbm_score,
                    "Extra_tree_Target": extra_trees_pred,
                    "bagging_Target": bagging_pred,
                    "Random_forest_Target": random_forest_pred,
                    "lgbm_prediction": lgbm_hla,
                    "extra_hla": ext_hla,
                    "hist_hla": hist_hla
                })

                score_df.to_csv("final_output.csv")
                score = pd.read_csv('final_output.csv')
                st.header("The File with score and values")
                st.write(score)

                print(score['Random_forest_Target'].values)
                print(score['Extra_tree_Target'].values)
                count_ones = score[['Extra_tree_Target', 'Random_forest_Target', 'bagging_Target']].sum(axis=1)
                score['Target'] = (count_ones > 2).astype(int)
                score.to_csv('target_final.csv')
                df_final = pd.read_csv('target_final.csv')
                df = df_final['extra_hla']
                amino_acid_values = {
                    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
                }

                def calculate_numerical_score(hla):
                    hla = str(hla)
                    alleles = hla.split('/')
                    total_score = 0
                    for allele in alleles:
                        amino_acid_sequence = ''.join(char for char in allele if char.isalpha())
                        numeric_multiplier = ''.join(char for char in allele if char.isdigit())
                        for amino_acid in amino_acid_sequence:
                            if numeric_multiplier:
                                total_score += abs(amino_acid_values.get(amino_acid, 0)) * float(numeric_multiplier)
                    return total_score

                def find_nearest_hla(total_score, hla_strings, k=10):
                    nearest_hla_list = []
                    hla_strings_sorted = sorted(hla_strings,
                                                key=lambda hla: abs(calculate_numerical_score(hla) - total_score))
                    for hla in hla_strings_sorted[:k]:
                        nearest_hla_list.append(hla)
                    return nearest_hla_list

                data = pd.read_csv('hla.csv')
                hla_output = []
                pred = []
                hla_strings = [str(hla) for hla in data['HLA'].values]
                for i in range(len(df_final)):
                    nearest_hla_list = find_nearest_hla(df_final['extra_hla'].values[i], hla_strings, k=10)
                    print("10 Nearest HLA Strings:")
                    for i, hla in enumerate(nearest_hla_list, start=1):
                        print(hla)
                        pred.append(hla)
                        if len(pred) == 10:
                            hla_output.append(pred)
                            pred = []
                df_final['hla_values'] = hla_output
                df_final.to_csv("expected.csv")
                df_d3 = pd.read_csv("expected.csv")
                target = df_d3[['Extra_tree_Target', 'Random_forest_Target']].sum(axis=1)
                print('---------------------------------------------')
                print(target)
                df_d3['Target'] = (target > 1).astype(int)
                print(df_d3.Target)
                print('-------------------------------------------------------')
                df_tar = df_d3[df_d3['Target'] == 1]
                if len(df_tar.Target.values) <= 30:
                    df_tar.to_csv('target.csv')
                    df_tab = pd.read_csv('target.csv')
                    print(df_tab.columns)
                    col = ['start', 'end', 'Epitope', 'hla_values']
                    st.table(df_tab[col])
                else:
                    values = df_d3['XGB_predicted_score'].sort_values(ascending=False).values
                    val = []
                    for i in range(len(values)):
                        val.append(values[i])
                        if len(val) == 30:
                            break
                    epitope = []
                    hla = []
                    starts = []
                    ends = []
                    for i in val:
                        df_val = df_d3[df_d3['XGB_predicted_score'] == i]
                        epitope.append(df_val.Epitope)
                        hla.append(df_val.hla_values)
                        starts.append(df_val.start)
                        ends.append(df_val.end)
                    data_dict = {
                        'Epitope': epitope,
                        'HLA': hla,
                        'Start': starts,
                        'End': ends
                    }
                    df = pd.DataFrame(data_dict)
                    df = df.explode('Epitope').explode('HLA').explode('Start').explode('End')
                    df.reset_index(drop=True, inplace=True)
                    print(df)
                    st.write(df)
    elif page == "About":
        st.title("About Us")
        st.write("""
    ## About This App
    In the quest to enhance cancer immunotherapy, epitope prediction plays a pivotal role in identifying potential targets for vaccine development. In this study, we present an integrated pipeline comprising three distinct models aimed at epitope prediction and prioritization for cancer immunotherapy. 

    ## Contact Information
    If you have any questions or feedback, please feel free to reach out to us:
    - Karthick Vasudevan
    - Email: karthick.1087@gmail.com
    - Address: Institute of Bioinformatics, Bengaluru
    - Dhanuskumar T
    - Email: danukumar517@gmail.com
    - Sripad Rama Hebbar
    - Email: sripadkarthik@gmail.com 
    """)

    # Help page content
    elif page == "Help":
        st.title("Help & Support")
        st.write("""
    ## Getting Started
    To get started with this app, follow these steps:
    Inputs Requirements:
    1. Tumor antigen Protein sequence
    2. Specify the type of MHC requirement (MHC-1 or MHC-2 or both)
    3. The given protein sequence will be chunked into peptides (probable epitopes) and feature of those peptides will be generated.
    4. The input protein sequence features will be extracted and results will be gnerated
    5. For more informartion refer https://github.com/karthick1087/VaxOptiML/blob/main/README.md
    """)


if __name__=="__main__":
    main()
