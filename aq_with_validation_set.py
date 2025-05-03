import numpy as np
import pandas as pd
import copy
from tabulate import tabulate
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Algorytm AQ do generowania reguł z pliku CSV"
    )
    parser.add_argument(
        "-f", "--file", type=str, required=True,
        help="Ścieżka do pliku CSV z danymi wejściowymi."
    )
    parser.add_argument(
        "-s", "--seed", type=str , required=True,
        help="Klasa pozytywna, która ma być analizowana."
    )
    parser.add_argument(
        "-r", "--training-set-ratio", type=float, required=True,
        help="Odsetek liczby przykładów treningowych z całego zbioru."
    )
    parser.add_argument(
        "-v", "--validation-set-ratio", type=float, required=True,
        help="Odsetek liczby przykładów walidacyjnych z całego zbioru."
    )
    parser.add_argument(
        "-vs", "--validation-sets-number", type=float, required=True,
        help="Liczba podzbiorów walidacyjnych."
    )
    parser.add_argument(
        "-m", type=int, required=True,
        help="Liczba m najlepszych kompleksów."
    )
    parser.add_argument(
        "-t", "--hamming-distance-ratio", type=float, required=True,
        help="Maksymalna odległość Hamminga między ziarnem pozytywnym a negatywnym względem liczby atrybutów."
    )
    return parser.parse_args()

def find_conflicts(df): # sprawdzanie konfliktów w datasecie
    label_column = 'class'
    feature_columns = df.columns.difference([label_column]).tolist() # uzyskiwanie nazw kolumn
    grouped = df.groupby(feature_columns)[label_column].nunique() # grupowanie wierszy o tych samych atrybutach i obliczanie liczby unikalnych klas

    conflicts = grouped[grouped > 1] # weryfikowanie czy istnieją konflikty

    if not conflicts.empty:
        conflict_keys = conflicts.index.tolist()
        conflicting_rows = []
        for index, row in df.iterrows():
            row_values = tuple(row[col] for col in feature_columns)
            if row_values in conflict_keys:
                conflicting_rows.append(row)

        conflicting_rows_table = pd.DataFrame(conflicting_rows)
        print(f"Liczba sprzecznych przykładów: {len(conflicts)}")
        print("\nSprzeczne wiersze:")
        print(conflicting_rows_table)
        exit(1)
    else:
        print("Brak sprzecznych wierszy.")

def split_dataset(dataset, r, v): # dzielenie datasetu na zbiór treningowy, walidacyjny i testowy
    split_index1 = int(r * len(dataset))
    split_index2 = int((r+v) * len(dataset))
    train_data = dataset[:split_index1]
    validation_data = dataset[split_index1:split_index2]
    test_data = dataset[split_index2:]
    return train_data, validation_data, test_data

def initialize_general_complex(df):
    general_rule = {}
    for attr in df.columns:
        if attr != 'class':  # pomijamy etykietę klasy
            general_rule[attr] = set(df[attr].unique())
    return [general_rule]  # zwracamy listę reguł (na start jedna)

def hamming_distance(pos_seed, neg_seed):
    distance = 0
    for attr in pos_seed.index:
        if attr != 'class':
            if pos_seed[attr] != neg_seed[attr]:
                distance += 1
    return distance

def positive_examples(df, seed): # zwraca dataset przykładów klasy pozytywnej
    # dopasuj typ argumentu seed do typu danych w kolumnie 'class'
    class_type = df['class'].dtype
    if pd.api.types.is_numeric_dtype(class_type):
        seed = int(seed)
    return df[df['class'] == seed]


def negative_examples(df, seed, t): # zwraca dataset przykładów klasy negatywnej
    class_type = df['class'].dtype
    if pd.api.types.is_numeric_dtype(class_type):
        seed = int(seed)
    num_attributes = df.shape[1] - 1
    threshold = t * num_attributes
    neg_examples = df[df['class'] != seed]
    filtered_examples = []
    pos_seed = df[df['class'] == seed].iloc[0]
    for _, neg_seed in neg_examples.iterrows():
        distance = hamming_distance(pos_seed, neg_seed)
        if distance <= threshold:
            filtered_examples.append(neg_seed)

    return pd.DataFrame(filtered_examples)

def negative_examples_validation(df, seed): # zwraca dataset przykładów klasy negatywnej
    class_type = df['class'].dtype
    if pd.api.types.is_numeric_dtype(class_type):
        seed = int(seed)
    return df[df['class'] != seed]

def return_positive_seed(pos_set):
    pos_set = list(pos_set.to_dict(orient='records'))
    return pos_set[0]



def return_negative_seed(neg_set):
    neg_set = list(neg_set.to_dict(orient='records'))
    return neg_set[0]



def covers(rule, example):
    '''
    funkcja pomocnicza do funkcji check_coverage_[...]_examples
    '''
    for attr in rule:
        if example[attr] not in rule[attr]:
            return False # nie pokrywa
    return True # pokrywa



def check_coverage_of_negative_examples(neg_set, complexes):
    filtered_rows = []
    for _, example in neg_set.iterrows():
        if any(covers(rule, example) for rule in complexes):
            filtered_rows.append(example)

    return pd.DataFrame(filtered_rows) # zwraca listę niewyeliminowanych negatywnych przykładów



def check_coverage_of_positive_examples(pos_set, complexes):
    filtered_rows = []
    for _, example in pos_set.iterrows():
        if not any(covers(rule, example) for rule in complexes):
            filtered_rows.append(example)

    return pd.DataFrame(filtered_rows) # zwraca listę jeszcze niepokrytych pozytywnych przykładów



def compare_seeds(pos_seed, neg_seed):
    '''
    funkcja porównuje różnice pomiędzy ziarnem xs i xn oraz zwraca atrybuty, które się różnią
    '''
    differing_attrs = []
    for i, attr in enumerate(pos_seed):
        if attr == 'class':
            continue
        if pos_seed[attr] != neg_seed[attr]:
            differing_attrs.append(attr)  # index + name
    return differing_attrs



def generate_complex(compared_seeds, neg_seed, complex):
    '''
    generuje nowe kompleksy zgodnie z zasadą gwiazdy
    '''
    final_complex = []

    for comp in range(len(complex)):

        for attr in range(len(compared_seeds)):
            temp_complex = copy.deepcopy(complex)
            val = temp_complex[comp][compared_seeds[attr]]

            val.discard(neg_seed[compared_seeds[attr]])

            final_complex.append(temp_complex[comp])
    return final_complex



def evaluate_complexes(complexes, positive_examples, negative_examples, m):
    '''
    funkcja oceny kompleksów określona jako liczba pokrywanych przez nie 
    przykładów o kategorii identycznej z kategorią ziarna nie pokrytych 
    przez wygenerowane wcześniej reguły
    '''
    def f1(complex):
        # Liczba pokrywanych pozytywnych przykładów
        return sum(covers(complex, ex) for _, ex in positive_examples.iterrows())

    def f2(complex):
        # Liczba NIEpokrywanych negatywnych przykładów
        return sum(not covers(complex, ex) for _, ex in negative_examples.iterrows())

    # Nadaj ocenę każdemu kompleksowi 
    scored = []
    for idx, c in enumerate(complexes):
        score = (f1(c) + f2(c)) # tutaj można dodać , f2(c)
        scored.append((score, idx, c))  # Dodaj indeks jako tie-breaker

    # Sortuj malejąco wg score, a przy remisie preferuj wyższy indeks (czyli późniejszy kompleks)
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))

    # Zwróć m najlepszych kompleksów
    best = [c for _, _, c in scored[:m]]
    return best



def print_rules(rules):
    """
    Funkcja wypisująca wszystkie utworzone reguły w formie tabeli,
    dynamicznie dostosowując się do nazw atrybutów.
    """
    if not rules:
        print("Brak reguł do wyświetlenia.")
        return

    # Zakładamy, że każda reguła to lista z jednym słownikiem
    attribute_names = list(rules[0][0].keys())
    headers = ["#"] + attribute_names + ["c(x)"]

    table = []
    for i, rule in enumerate(rules, 1):
        rule_dict = rule[0]  # Zakładamy, że complex to [reguła]
        row = [i]
        for attr in attribute_names:
            row.append(" ∨ ".join(sorted(rule_dict.get(attr, []))))
        row.append("0")  # Można dodać oznaczenie klasy lub c(x)
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid"))



def check_aq_algorithm(neg_set, pos_set, complexes):
    x = check_coverage_of_negative_examples(neg_set, complexes)

    y = check_coverage_of_positive_examples(pos_set, complexes)

    if x.empty:
        print('Wszystkie negatywne przykłady są wykluczone')
    else:
        print(f'Niewykluczone przykłady negatywne (FP): {len(x)}')

    if y.empty:
        print('Wszystkie pozytywne przykłady zawierają się w zestawie reguł')
    else:
        print(f'Niepokryte przykłady pozytywne (FN): {len(y)}')

    return len(x), len(y)

def accuracy(FN, FP, dataset_size):
    return round(((dataset_size - (FN + FP)) / dataset_size) * 100, 2)

def recall(TP, FN):
    return round(TP / (TP + FN) * 100, 2)

def precision(TP, FP):
    return round(TP / (TP + FP) * 100, 2)

def main():
    '''
    główne wykonanie programu
    '''
    
    args = parse_arguments()
    df = pd.read_csv(args.file)
    find_conflicts(df)
    training_dataset, validation_dataset, test_dataset = split_dataset(df, args.training_set_ratio, args.validation_set_ratio)
    seed = args.seed
    validation_sets_number = args.validation_sets_number
    validation_sets = np.array_split(validation_dataset, validation_sets_number)
    print(validation_sets[0])

    print("\n" + "-" * 80 + "    OBLICZENIA    " + "-" * 80 + "\n")

    neg_set_mark = negative_examples(training_dataset, seed, args.hamming_distance_ratio) # zestaw wszystkich negatywnych przykładów
    pos_set = positive_examples(training_dataset, seed) # zestaw pozytywnych przykładów, który jest aktualizowany po iteracji poprzez usunięcie pokrytych przykładów
    pos_set_mark = pos_set

    neg_set_mark_test = negative_examples(test_dataset, seed, args.hamming_distance_ratio) # zestaw wszystkich negatywnych przykładów
    pos_set_test = positive_examples(test_dataset, seed) # zestaw pozytywnych przykładów, który jest aktualizowany po iteracji poprzez usunięcie pokrytych przykładów
    pos_set_mark_test = pos_set_test

    set_of_rules = []

    while not pos_set.empty : # dopóki zbiór reguł nie pokrywa wszystkich przykładów
        
        complex = initialize_general_complex(training_dataset)
        pos_seed = return_positive_seed(pos_set)
        print('pos seed')
        print(pos_seed)
 
        neg_set = neg_set_mark
        i = 0
        while not neg_set.empty : # dopóki zbiór negatywny nie jest pusty
            neg_seed = return_negative_seed(neg_set)

            compared_seeds = compare_seeds(pos_seed, neg_seed)

            complex = generate_complex(compared_seeds, neg_seed, complex)

            neg_set = check_coverage_of_negative_examples(neg_set,complex)

            pos_val_set = positive_examples(validation_sets[i], seed)
            neg_val_set = negative_examples_validation(validation_sets[i], seed)
            complex = evaluate_complexes(complex, pos_val_set, neg_val_set, args.m) # ocena
            i = i + 1

        complex = evaluate_complexes(complex, pos_set, neg_set_mark, args.m)
        print('Dodanie zasady do set of rules')
        print(complex)
        set_of_rules.extend(complex)
        pos_set = check_coverage_of_positive_examples(pos_set, complex) # zwraca niepokryte przykłady
        print('Zbiór pozytywnych przykładów, które pozostały :')
        print(pos_set)
    
   # print_rules(set_of_rules)
    print("\n" + "-" * 80 + "    UZYSKANE REGUŁY    " + "-" * 80 + "\n")
    print(set_of_rules)

    print("\n" + "-" * 80 + "    LICZBA REGUŁ    " + "-" * 80 + "\n")
    print(f'Liczba zasad : {len(set_of_rules)}')

    print("\n" + "-" * 75 + "    TESTY POKRYCIA    " + "-" * 75 + "\n")
    check_aq_algorithm(neg_set_mark, pos_set_mark, set_of_rules)

    print("\n" + "-" * 70 + "    TESTY NA ZBIORZE TESTOWYM   " + "-" * 70 + "\n")
    FP, FN = check_aq_algorithm(neg_set_mark_test, pos_set_mark_test, set_of_rules)
    print("\nWYNIKI:")
    print(f"Dokładność: {accuracy(FN, FP, len(test_dataset))}%")
    print(f"Czułość: {recall(len(pos_set_mark_test) - FN, FN)}%")
    print(f"Precyzja: {recall(len(pos_set_mark_test) - FN, FP)}%")

main()




