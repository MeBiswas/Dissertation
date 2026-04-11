base_path = r"..\data\raw"

# RESULTS PATH
schcs_results_path = r"..\data\sch_cs"
preprocessed_results_path = r"..\data\preprocessed"
chm_corrected_results_path = r"..\data\chm_corrected"
phi_initialized_result_path = r"..\data\phi_initialized"
level_set_iterated_results_path = r"..\data\level_set_iterated"

# DATASET PATHS
dmr_ir_o = r"\DMR-IR-O"
bcd_dataset = {
    "sick": r"\BCD_Dataset\Sick",
    "healthy": r"\BCD_Dataset\normal",
    "unknown": r"\BCD_Dataset\Unknown_class"
}
breast_thermography_dataset = {
    "sick": r"\Breast Thermography\Malignant\IIR0119",
    "healthy": r"\Breast Thermography\Benign\IIR0118"
}
breast_cancer_dataset = {
    "sick": r"\breast-cancer-dataset\Train\Malignant",
    "healthy": r"\breast-cancer-dataset\Train\Benign"
}
dmr_ir_diff_view_dataset = {
    "sick": r"\DMR_IR_Different_View\Sick\0256",
    "healthy": r"\DMR_IR_Different_View\Healthy\0259"
}
thiago_dataset = {
    "healthy": r"\Benign\226\Segmentadas",
    "diseased": r"\Malignant\255\Segmentadas",
    "mat_diseased": r"\Malignant\255\Matrizes\PAC_54_DN19.txt",
    "testing": r"\Images and Matrices from the Thesis of Thiago Alves Elias da Silva\12 New Test Cases - Testing",
    "training": r"\Images and Matrices from the Thesis of Thiago Alves Elias da Silva\Methodology Development - Training"
}