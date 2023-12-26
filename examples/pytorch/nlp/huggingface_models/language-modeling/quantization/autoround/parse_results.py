
import sys
import json
import pprint

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
    mmlu_results = {}
    other_results = {}
    for task in data.keys():
        acc_dict = {"Hums": [0, 0., 0.], "STEM": [0, 0., 0.], "Social": [0, 0., 0.], "Other": [0, 0., 0.],
                    "Avg.": [0, 0., 0.]}
        if isinstance(data[task], float):
            other_results[task] = data[task]
        else:
            results = data[task]['results']
            if "hendrycksTest" in task:
                for sub_task in results.keys():
                    sub_name = sub_task.split("hendrycksTest-")[1]
                    domain = task_domain[sub_name][0]
                    domain_list = acc_dict[domain]
                    domain_list[0] += 1
                    domain_list[1] += results[sub_task]["acc"]
                    domain_list[2] += results[sub_task]["acc_norm"]
                    acc_dict['Avg.'][0] += 1
                    acc_dict['Avg.'][1] += results[sub_task]["acc"]
                    acc_dict['Avg.'][2] += results[sub_task]["acc_norm"]

                mmlu_results[task] = {}
                for key in acc_dict.keys():
                    mmlu_results[task][key] = {}
                    mmlu_results[task][key]['acc'] = acc_dict[key][1] / acc_dict[key][0]
                    mmlu_results[task][key]['acc_norm'] = acc_dict[key][2] / acc_dict[key][0]
            else:
                other_results[task] = results

    # other_results = {}
    # for key in results.keys():
    #     if 'hendrycksTest' in key:
    #         sub_name = key.split("hendrycksTest-")[1]
    #         domain=task_domain[sub_name][0]
    #         domain_list = acc_dict[domain]
    #         domain_list[0] += 1
    #         domain_list[1] += results[key]["acc"]
    #         domain_list[2] += results[key]["acc_norm"]
    #         acc_dict['Avg.'][0] += 1
    #         acc_dict['Avg.'][1] += results[key]["acc"]
    #         acc_dict['Avg.'][2] += results[key]["acc_norm"]
    #     else:
    # other_results[key] = results[key]

    # mmlu_results = {}
    # for key in acc_dict.keys():
    #     mmlu_results[key] = {}
    #     mmlu_results[key]['acc'] = acc_dict[key][1]/acc_dict[key][0]
    #     mmlu_results[key]['acc_norm'] = acc_dict[key][2]/acc_dict[key][0]

    print("***** MMLU RESULTS *****")
    pprint.pprint(mmlu_results)
    print()

    print("***** OTHER RESULTS *****")
    pprint.pprint(other_results)
    print()
    new_results = mmlu_results
    new_results.update(other_results)
    return new_results






