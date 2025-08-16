import pandas as pd
import numpy as np

# --- Load Your Data ---
# Make sure 'clinical_data.csv' and 'mutation_data.csv' are in the same folder as this script
try:
    clinical_df = pd.read_csv('clinical_data.csv')
    mutation_df = pd.read_csv('mutation_data.csv')
except FileNotFoundError:
    print("Error: Make sure 'clinical_data.csv' and 'mutation_data.csv' are in the correct directory.")
    exit()


# --- Skewing Clinical Data ---

# Create a new DataFrame for the skewed data
skewed_clinical_df = pd.DataFrame()

# Regenerate 'Patient_ID' and 'Sample_ID'
num_samples = len(clinical_df)
skewed_clinical_df['Patient_ID'] = [f'TCGA-SKEWED-{i:06d}' for i in range(num_samples)]
skewed_clinical_df['Sample_ID'] = [f'TCGA-SKEWED-{i:06d}-01' for i in range(num_samples)]

# Skew 'Age_at_Diagnosis' (positively skewed)
skewed_clinical_df['Age_at_Diagnosis'] = np.random.gamma(shape=10, scale=6, size=num_samples).astype(int)

# Skew 'Gender' (e.g., more females)
skewed_clinical_df['Gender'] = np.random.choice(['Female', 'Male'], size=num_samples, p=[0.65, 0.35])

# Skew 'Race' (e.g., more White participants)
skewed_clinical_df['Race'] = np.random.choice(['White', 'Asian', 'Black or African American', 'Other'], size=num_samples, p=[0.75, 0.15, 0.05, 0.05])

# Skew 'Cancer_Type' and 'Subtype'
cancer_types = ['Breast Invasive Carcinoma', 'Colon Adenocarcinoma', 'Lung Adenocarcinoma', 'Glioblastoma Multiforme']
cancer_probs = [0.4, 0.3, 0.2, 0.1]
skewed_clinical_df['Cancer_Type'] = np.random.choice(cancer_types, size=num_samples, p=cancer_probs)

subtypes = {
    'Breast Invasive Carcinoma': ['Luminal A', 'HER2-enriched', 'Triple-negative'],
    'Colon Adenocarcinoma': ['Adenocarcinoma', 'Mucinous Adenocarcinoma'],
    'Lung Adenocarcinoma': ['Acinar', 'Papillary', 'Solid'],
    'Glioblastoma Multiforme': ['Classical', 'Mesenchymal', 'Proneural']
}
skewed_clinical_df['Subtype'] = skewed_clinical_df['Cancer_Type'].apply(lambda x: np.random.choice(subtypes.get(x, ['N/A'])))


# Skew 'Stage;' and 'Grade'
skewed_clinical_df['Stage'] = np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], size=num_samples, p=[0.25, 0.35, 0.25, 0.15])
skewed_clinical_df['Grade'] = np.random.choice([1, 2, 3], size=num_samples, p=[0.2, 0.5, 0.3])


# Skew hormone receptor status
skewed_clinical_df['ER_Status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.7, 0.3])
skewed_clinical_df['PR_Status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.6, 0.4])
skewed_clinical_df['HER2_Status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.2, 0.8])

# Skew 'Family_History' and 'Smoking_Status'
skewed_clinical_df['Family_History'] = np.random.choice(['No', 'Yes'], size=num_samples, p=[0.8, 0.2])
skewed_clinical_df['Smoking_Status'] = np.random.choice(['Never', 'Former', 'Current'], size=num_samples, p=[0.5, 0.35, 0.15])


# Skew 'Treatment_Type'
skewed_clinical_df['Treatment_Type'] = np.random.choice(['Surgery only', 'Surgery+Chemo', 'Chemo+Radiotherapy', 'Targeted Therapy'], size=num_samples, p=[0.3, 0.4, 0.2, 0.1])


# Skew 'Survival_Time_days' and 'Vital_Status'
def generate_survival_time(stage):
    if stage == 'Stage I':
        return int(np.random.weibull(2) * 4000)
    elif stage == 'Stage II':
        return int(np.random.weibull(2) * 3000)
    elif stage == 'Stage III':
        return int(np.random.weibull(1.5) * 2000)
    else: # Stage IV
        return int(np.random.weibull(1.2) * 1000)

skewed_clinical_df['Survival_Time_days'] = skewed_clinical_df['Stage'].apply(generate_survival_time)
skewed_clinical_df['Vital_Status'] = np.where(skewed_clinical_df['Survival_Time_days'] < 1825, 'Deceased', 'Alive') # 5 years = 1825 days


# Save the skewed clinical data to a new CSV file
skewed_clinical_df.to_csv('skewed_clinical_data.csv', index=False)
print("Successfully generated 'skewed_clinical_data.csv'")


# --- Skewing Mutation Data ---

# Create a new DataFrame for skewed mutation data
skewed_mutation_df = pd.DataFrame()

# Regenerate mutation data for the new samples
new_sample_ids = skewed_clinical_df['Sample_ID'].tolist()
num_mutations = len(mutation_df)
skewed_mutation_df['Sample_ID'] = np.random.choice(new_sample_ids, size=num_mutations, replace=True)

# Skew 'Hugo_Symbol'
common_genes = ['TP53', 'PIK3CA', 'GATA3', 'BRCA1', 'BRCA2', 'KRAS', 'EGFR']
gene_probs = np.array([0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1])
gene_probs /= gene_probs.sum()

other_genes = [g for g in mutation_df['Hugo_Symbol'].unique() if g not in common_genes]

p_common = 0.8
n_common = int(num_mutations * p_common)
n_other = num_mutations - n_common

# --- FIXED SECTION ---
# Check if other_genes list is empty before trying to sample from it
if other_genes:
    skewed_genes = list(np.random.choice(common_genes, size=n_common, p=gene_probs)) + \
                   list(np.random.choice(other_genes, size=n_other, replace=True))
else:
    # If there are no "other" genes, get all samples from the "common" list
    skewed_genes = list(np.random.choice(common_genes, size=num_mutations, p=gene_probs, replace=True))

np.random.shuffle(skewed_genes)
skewed_mutation_df['Hugo_Symbol'] = skewed_genes
# --- END FIXED SECTION ---


# Skew 'Variant_Classification'
variant_classes = ['Missense_Mutation', 'Nonsense_Mutation', 'Frameshift_Ins', 'Frameshift_Del', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins']
variant_probs = [0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
skewed_mutation_df['Variant_Classification'] = np.random.choice(variant_classes, size=num_mutations, p=variant_probs)

skewed_mutation_df['Variant_Type'] = skewed_mutation_df['Variant_Classification'].apply(lambda x: 'SNP' if 'Missense' in x or 'Nonsense' in x else ('DEL' if 'Del' in x else 'INS'))

# Skew 'Allele_Frequency'
skewed_mutation_df['Allele_Frequency'] = np.random.beta(a=2, b=5, size=num_mutations)

# Fill in other columns
skewed_mutation_df['Entrez_Gene_ID'] = np.nan
skewed_mutation_df['Chromosome'] = np.random.randint(1, 23, size=num_mutations)
skewed_mutation_df['Start_Position'] = np.random.randint(100000, 200000000, size=num_mutations)
skewed_mutation_df['End_Position'] = skewed_mutation_df['Start_Position'] + np.random.randint(1, 100)
skewed_mutation_df['Reference_Allele'] = np.random.choice(['A', 'C', 'G', 'T'], size=num_mutations)
skewed_mutation_df['Tumor_Seq_Allele'] = np.random.choice(['A', 'C', 'G', 'T'], size=num_mutations)
skewed_mutation_df['Protein_Change'] = 'p.X' # Corrected variable name here
skewed_mutation_df['Mutation_Status'] = 'Somatic'

# Reorder columns
original_mutation_columns = ['Hugo_Symbol', 'Entrez_Gene_ID', 'Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele', 'Variant_Type', 'Variant_Classification', 'Protein_Change', 'Sample_ID', 'Mutation_Status', 'Allele_Frequency', 'End_Position']
skewed_mutation_df = skewed_mutation_df.reindex(columns=original_mutation_columns)


# Save the skewed mutation data
skewed_mutation_df.to_csv('skewed_mutation_data.csv', index=False)
print("Successfully generated 'skewed_mutation_data.csv'")