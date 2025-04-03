import json
import os
import csv
import random


def convert_dataset(dataset_name, directory):
    # Construct input file paths.
    term2def_path = os.path.join(directory, "term2def.csv")
    taxo_path = os.path.join(directory, f"{dataset_name}.taxo")

    # Construct output file paths.
    dic_json_path = os.path.join(directory, "dic.json")
    raw_taxo_path = os.path.join(directory, f"{dataset_name}_raw_en.taxo")
    train_taxo_path = os.path.join(directory, f"{dataset_name}_train.taxo")
    eval_gt_path = os.path.join(directory, f"{dataset_name}_eval.gt")
    eval_terms_path = os.path.join(directory, f"{dataset_name}_eval.terms")

    # Load term definitions from term2def.csv
    dic = {}
    id2term = {}
    with open(term2def_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 4:
                term_id = row[1].strip()
                term = row[2].strip()
                definition = row[3].strip()
                dic[term] = [definition]
                id2term[term_id] = term
            else:
                term = row[0].strip()
                definition = row[2].strip()
                dic[term] = [definition]

    # Read the taxonomy file: each line is expected as "child<TAB>parent".
    taxonomy = {}
    reverse_taxonomy = {}
    with open(taxo_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                child_term = parts[0].strip()
                parent_term = parts[1].strip()
                try:
                    child_term = id2term[child_term]
                    parent_term = id2term[parent_term]
                except:
                    pass
                reverse_taxonomy[child_term] = reverse_taxonomy.get(child_term, []) + [
                    parent_term
                ]
                taxonomy[parent_term] = taxonomy.get(parent_term, []) + [child_term]

    # store nodes with no parents (roots)
    roots = [term for term in dic.keys() if len(reverse_taxonomy.get(term, [])) == 0]
    taxonomy[dataset_name] = roots
    for root in roots:
        reverse_taxonomy[root] = [dataset_name]
    # find nodes with a single parent to use as evaluation set
    eval_terms = [
        k
        for k, v in reverse_taxonomy.items()
        if len(v) == 1 and v[0] != dataset_name and len(taxonomy.get(v[0], [])) == 0
    ]
    random.shuffle(eval_terms)
    eval_terms = eval_terms[: int(len(eval_terms) // 5)]

    # Build dic.json: map each term to a list with its definition if available.
    dic[dataset_name] = [""]  # Add the dataset name to the dictionary.
    with open(dic_json_path, "w", encoding="utf-8") as f:
        json.dump(dic, f, indent=4, ensure_ascii=False)

    # Write {dataset}_raw_en.taxo: add index column.
    with open(raw_taxo_path, "w", encoding="utf-8") as f:
        for idx, (parent, children) in enumerate(taxonomy.items()):
            for child in children:
                f.write(f"{idx}\t{child}\t{parent}\n")

    # Write {dataset}_train.taxo: swap order (parent first, then child).
    with open(train_taxo_path, "w", encoding="utf-8") as f:
        for parent, children in taxonomy.items():
            for child in children:
                if child not in eval_terms and parent not in eval_terms:
                    f.write(f"{parent}\t{child}\n")

    # Write {dataset}_eval.gt: list parent terms (one per line).
    with open(eval_gt_path, "w", encoding="utf-8") as f:
        for term in eval_terms:
            parent = reverse_taxonomy[term][0]
            f.write(f"{parent}\n")

    # Write {dataset}_eval.terms: list child terms (one per line).
    with open(eval_terms_path, "w", encoding="utf-8") as f:
        for term in eval_terms:
            f.write(f"{term}\n")

    print("Conversion complete. Output files created in", directory)


convert_dataset("SemEval-Verb", "SemEval-Verb")
convert_dataset("SemEval-Noun", "SemEval-Noun")
convert_dataset("MAG-CS-Wiki", "MAG-CS-Wiki")
convert_dataset("MAG-PSY", "MAG-PSY")
