import sys
import json

# input_file = sys.argv[1]

# with open(input_file) as f:
#     data = json.load(f)
    
# data = data["results"]

task_domain = {
        "abstract_algebra": ["STEM"],
        "anatomy": ["Other"],
        "astronomy": ["STEM"],
        "business_ethics": ["Other"],
        "clinical_knowledge": ["Other"],
        "college_biology": ["STEM"],
        "college_chemistry": ["STEM"],
        "college_computer_science": ["STEM"],
        "college_mathematics": ["STEM"],
        "college_medicine": ["Other"],
        "college_physics": ["STEM"],
        "computer_security": ["STEM"],
        "conceptual_physics": ["STEM"],
        "econometrics": ["Social"],
        "electrical_engineering": ["STEM"],
        "elementary_mathematics": ["STEM"],
        "formal_logic": ["Hums"],
        "global_facts": ["Other"],
        "high_school_biology": ["STEM"],
        "high_school_chemistry": ["STEM"],
        "high_school_computer_science": ["STEM"],
        "high_school_european_history": ["Hums"],
        "high_school_geography": ["Social"],
        "high_school_government_and_politics": ["Social"],
        "high_school_macroeconomics": ["Social"],
        "high_school_mathematics": ["STEM"],
        "high_school_microeconomics": ["Social"],
        "high_school_physics": ["STEM"],
        "high_school_psychology": ["Social"],
        "high_school_statistics": ["STEM"],
        "high_school_us_history": ["Hums"],
        "high_school_world_history": ["Hums"],
        "human_aging": ["Other"],
        "human_sexuality": ["Social"],
        "international_law": ["Hums"],
        "jurisprudence": ["Hums"],
        "logical_fallacies": ["Hums"],
        "machine_learning": ["STEM"],
        "management": ["Other"],
        "marketing": ["Other"],
        "medical_genetics": ["Other"],
        "miscellaneous": ["Other"],
        "moral_disputes": ["Hums"],
        "moral_scenarios": ["Hums"],
        "nutrition": ["Other"],
        "philosophy": ["Hums"],
        "prehistory": ["Hums"],
        "professional_accounting": ["Other"],
        "professional_law": ["Hums"],
        "professional_medicine": ["Other"],
        "professional_psychology": ["Social"],
        "public_relations": ["Social"],
        "security_studies": ["Social"],
        "sociology": ["Social"],
        "us_foreign_policy": ["Social"],
        "virology": ["Other"],
        "world_religions": ["Hums"],
    }

def result_parser(data):
    results = data['results']
    acc_dict = {"Hums":[0,0.], "STEM":[0,0.], "Social":[0,0.], "Other":[0,0.], "Avg.":[0,0.]}

    for key in task_domain.keys():
        task_name=f"hendrycksTest-{key}"
        domain=task_domain[key][0]
        domain_list = acc_dict[domain]
        domain_list[0] += 1
        domain_list[1] += results[task_name]["acc"]
        acc_dict['Avg.'][0] += 1
        acc_dict['Avg.'][1] += results[task_name]["acc"]

    print("***** MMLU RESULTS *****")

    for key in acc_dict.keys():
        domain_acc = acc_dict[key][1]/acc_dict[key][0]
        print(f"{key}: {domain_acc}")
        print()
    
    for key in results.keys():
        if 'hendrycksTest' in key:
            continue
        print(f"{key}: {results[key]}")
        print()
    

