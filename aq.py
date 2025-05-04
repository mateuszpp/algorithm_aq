import pandas as pd
import copy
from tabulate import tabulate
import argparse

def parse_arguments(): # pobranie argumentów z konsoli
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
        "-rt", "--testing-set-ratio", type=float, required=True,
        help="Odsetek liczby przykładów testowych z całego zbioru."
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

        conflicting_rows_table = pd.DataFrame(conflicting_rows) # generowanie tabeli ze sprzecznymi rekordami
        print(f"Liczba sprzecznych przykładów: {len(conflicts)}")
        print("\nSprzeczne wiersze:")
        print(conflicting_rows_table)
        exit(1)
    else:
        print("Brak sprzecznych wierszy.")

def split_dataset(dataset, r, rt): # dzielenie datasetu na zbiór treningowy i testowy
    split_index = int(r * len(dataset)) # obliczenie, w którym miejscu ma nastąpić podział zbioru
    split_index2 = int((r + rt) * len(dataset))  # obliczenie, w którym miejscu ma nastąpić podział zbioru testowego
    train_data = dataset[:split_index] # utworzenie datasetu treningowego
    test_data = dataset[split_index:split_index2] # utworzenie datasetu testowego
    return train_data, test_data

def initialize_general_complex(df): # generowanie najbardziej ogólnej reguły
    general_rule = {}
    for attr in df.columns:
        if attr != 'class':  # pominięcie etykiety klasy
            general_rule[attr] = set(df[attr].unique()) # dodanie wszystkich unikatowych wartości atrybutów
    return [general_rule]  # zwracanie listy reguł (na start jedna)

def hamming_distance(pos_seed, neg_seed): # obliczanie odległości Hamminga między ziarnem pozytywnym i negatywnym
    distance = 0
    for attr in pos_seed.index:
        if attr != 'class':
            if pos_seed[attr] != neg_seed[attr]:
                distance += 1 # zwiększanie odległości dla każdego różniącego się wartością atrybutu
    return distance

def positive_examples(df, seed): # zwracanie datasetu przykładów klasy pozytywnej
    class_type = df['class'].dtype # dopasowanie typu argumentu seed do typu danych w kolumnie 'class'
    if pd.api.types.is_numeric_dtype(class_type):
        seed = int(seed)
    return df[df['class'] == seed]


def negative_examples(df, seed, t): # zwracanie dataset przykładów klasy negatywnej z uzwględnieniem odległości Hamminga
    class_type = df['class'].dtype # dopasowanie typu argumentu seed do typu danych w kolumnie 'class'
    if pd.api.types.is_numeric_dtype(class_type):
        seed = int(seed)
    num_attributes = df.shape[1] - 1 # obliczenie liczby atrybutów bez etykiety
    threshold = t * num_attributes # obliczenie maksymalnej odległości Hamminga między ziarnem pozytywnym i negatywnym
    neg_examples = df[df['class'] != seed]
    filtered_examples = []
    pos_seed = df[df['class'] == seed].iloc[0] # uzyskanie ziarna pozytywnego
    for _, neg_seed in neg_examples.iterrows():
        distance = hamming_distance(pos_seed, neg_seed) # obliczenie odległości Hamminga dla każdego ziarna negatywnego
        if distance <= threshold:
            filtered_examples.append(neg_seed) # dodanie ziarna negatywnego do listy, gdy jego odległość Hamminga od ziarna pozytywnego jest niewiększa niż podana przez użytkownika
    return pd.DataFrame(filtered_examples)

def return_positive_seed(pos_set): # zwrócenie ziarna pozytywnego
    pos_set = list(pos_set.to_dict(orient='records'))
    return pos_set[0]

def return_negative_seed(neg_set): # zwrócenie ziarna negatywnego
    neg_set = list(neg_set.to_dict(orient='records'))
    return neg_set[0]

def covers(rule, example): # funkcja pomocnicza do funkcji check_coverage...
    for attr in rule:
        if example[attr] not in rule[attr]: # sprawdzanie pokrycia atrybutów przykładu przez regułę
            return False # nie pokrywa
    return True # pokrywa

def check_coverage_of_negative_examples(neg_set, complexes): # sprawdzanie pokrycia przykładów klasy negatywnej
    filtered_rows = []
    for _, example in neg_set.iterrows(): # sprawdzanie pokrycia każdego przykładu przez reguły
        if any(covers(rule, example) for rule in complexes):
            filtered_rows.append(example)
    return pd.DataFrame(filtered_rows) # zwracanie listy niewyeliminowanych negatywnych przykładów

def check_coverage_of_positive_examples(pos_set, complexes): # sprawdzanie pokrycia przykładów klasy pozytywnej
    filtered_rows = []
    for _, example in pos_set.iterrows():
        if not any(covers(rule, example) for rule in complexes):
            filtered_rows.append(example)
    return pd.DataFrame(filtered_rows) # zwracanie listy jeszcze niepokrytych pozytywnych przykładów

def compare_seeds(pos_seed, neg_seed): # porównanie ziaren pozytywnych i negatywnych oraz zwrócenie listy różniących się atrybutów
    differing_attrs = []
    for i, attr in enumerate(pos_seed):
        if attr == 'class':
            continue
        if pos_seed[attr] != neg_seed[attr]:
            differing_attrs.append(attr)  # index + name
    return differing_attrs

def generate_complex(compared_seeds, neg_seed, complex): # generowanie kompleksów zgodnie z zasadą gwiazdy
    final_complex = []
    for comp in range(len(complex)): # iterowane po wszystkich kompleksach
        for attr in range(len(compared_seeds)): # iterowanie po wszystkich atrybutach o wartościach różniących się
            temp_complex = copy.deepcopy(complex) # kopiowanie oryginalnego kompleksu
            val = temp_complex[comp][compared_seeds[attr]] # pobranie wartości atrybutów
            val.discard(neg_seed[compared_seeds[attr]]) # usunięcie z kompleksu wartości przykładu negatywnego danego atrybutu
            final_complex.append(temp_complex[comp])
    return final_complex

def evaluate_complexes(complexes, positive_examples, negative_examples, m): # ocenianie kompleksów
    def f1(complex): # zliczanie liczby pokrywanych przykładów pozytywnych
        return sum(covers(complex, ex) for _, ex in positive_examples.iterrows())
    def f2(complex): # zliczanie liczby niepokrywanych przykładów negatywnych
        return sum(not covers(complex, ex) for _, ex in negative_examples.iterrows())

    scored = []
    for idx, c in enumerate(complexes):  # nadanie oceny każdemu kompleksowi
        score = (f1(c) + f2(c))
        scored.append((score, idx, c)) # dodanie indeksu jako tie-breaker

    scored.sort(reverse=True, key=lambda x: (x[0], x[1])) # sortowanie malejąco wg score, a przy remisie preferowanie wyższego indeksu (czyli późniejszego kompleksu)

    best = [c for _, _, c in scored[:m]] # zwrócenie m najlepszych kompleksów
    return best

def print_rules(rules): # wypisanie utworzonych reguł w formie tabeli
    if not rules:
        print("Brak reguł do wyświetlenia.")
        return

    attribute_names = list(rules[0][0].keys()) # każda reguła to lista z jednym słownikiem
    headers = ["#"] + attribute_names + ["c(x)"] # tworzenie nagłówków kolumn

    table = []
    for i, rule in enumerate(rules, 1):
        rule_dict = rule[0]
        row = [i]
        for attr in attribute_names:
            row.append(" ∨ ".join(sorted(rule_dict.get(attr, [])))) # dla każdego atrybutu pobieranie możliwych wartości i oddzielenie ich przez 'v'
        row.append("0")
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid"))

def check_aq_algorithm(neg_set, pos_set, complexes): # sprawdzanie pokrycia zbioru przykładów negatywnych i pozytywnych przez wygenerowanie reguły
    covered_neg = check_coverage_of_negative_examples(neg_set, complexes)
    not_covered_pos = check_coverage_of_positive_examples(pos_set, complexes)

    if covered_neg.empty:
        print('Wszystkie negatywne przykłady są wykluczone')
    else:
        print(f'Niewykluczone przykłady negatywne (FP): {len(covered_neg)}')

    if not_covered_pos.empty:
        print('Wszystkie pozytywne przykłady zawierają się w zestawie reguł')
    else:
        print(f'Niepokryte przykłady pozytywne (FN): {len(not_covered_pos)}')

    return len(covered_neg), len(not_covered_pos)

def accuracy(FN, FP, dataset_size): # obliczenie dokładności
    return round(((dataset_size - (FN + FP)) / dataset_size) * 100, 2)

def recall(TP, FN): # obliczenie czułości
    return round(TP / (TP + FN) * 100, 2)

def precision(TP, FP): # obliczenie precyzji
    return round(TP / (TP + FP) * 100, 2)

def main():
    args = parse_arguments() # pobranie argumentów z konsoli
    df = pd.read_csv(args.file).sample(frac = 1) # odczytanie i przetasowanie wierszy datasetu
    find_conflicts(df) # szukanie konfliktów w datasecie
    training_dataset, test_dataset = split_dataset(df, args.training_set_ratio, args.testing_set_ratio) # podzielenie datasetu na zbiór treningowy i testowy
    seed = args.seed # pobranie etykiety ziarna pozytywnego z konsoli

    print("\n" + "-" * 80 + "    OBLICZENIA    " + "-" * 80 + "\n")

    neg_set_mark = negative_examples(training_dataset, seed, args.hamming_distance_ratio) # stworzenie zestawu wszystkich negatywnych przykładów zbioru treningowego
    pos_set = positive_examples(training_dataset, seed) # stworzenie zestawu pozytywnych przykładów zbioru treningowego, który jest aktualizowany po iteracji poprzez usunięcie pokrytych przykładów
    pos_set_mark = pos_set # stworzenie zestawu pozytywnych przykładów zbioru treningowego

    set_of_rules = []

    while not pos_set.empty : # wykonywanie pętli dopóki zbiór reguł nie pokrywa wszystkich przykładów
        complex = initialize_general_complex(training_dataset) # wygenerwanie najbardziej ogólnej reguły
        pos_seed = return_positive_seed(pos_set) # pobranie ziarna pozytywnego
        neg_set = neg_set_mark # pobranie zbioru wszystkich przykładów negatywnych

        while not neg_set.empty : # wykonywanie pętli dopóki zbiór przykładów negatywnych nie jest pusty
            neg_seed = return_negative_seed(neg_set) # pobranie ziarna negatywnego
            compared_seeds = compare_seeds(pos_seed, neg_seed) # porównanie ziarna pozytywnego i negatynwgo
            complex = generate_complex(compared_seeds, neg_seed, complex) # wygenerowanie kompleksów
            neg_set = check_coverage_of_negative_examples(neg_set,complex) # zwrócenie nowej listy niewyeliminowanych przykładów
            complex = evaluate_complexes(complex, pos_set, neg_set_mark, args.m) # ocena kompleksów i zwrócenie m najlepszych

        complex = evaluate_complexes(complex, pos_set, neg_set_mark, args.m)  # ocena kompleksów i zwrócenie m najlepszych
        print('Dodanie zasady do set of rules: ')
        print(complex)
        set_of_rules.extend(complex) # dodanie do zbioru reguł
        pos_set = check_coverage_of_positive_examples(pos_set, complex) # zwrócenie niepokrytych przykładów klasy pozytywnej
        print('Zbiór pozytywnych przykładów, które pozostały:')
        print(pos_set)
    
    print("\n" + "-" * 80 + "    UZYSKANE REGUŁY    " + "-" * 80 + "\n")
    print(set_of_rules)

    print("\n" + "-" * 80 + "    LICZBA REGUŁ    " + "-" * 80 + "\n")
    print(f'Liczba zasad : {len(set_of_rules)}')

    print("\n" + "-" * 75 + "    TESTY POKRYCIA    " + "-" * 75 + "\n")
    check_aq_algorithm(neg_set_mark, pos_set_mark, set_of_rules)

    print("\n" + "-" * 70 + "    TESTY NA ZBIORZE TESTOWYM   " + "-" * 70 + "\n")
    neg_set_mark_test = negative_examples(test_dataset, seed, 1)  # stworzenie zestawu wszystkich negatywnych przykładów zbioru testowego
    pos_set_mark_test = positive_examples(test_dataset, seed) # stworzenie zestawu pozytywnych przykładów zbioru testowego

    FP, FN = check_aq_algorithm(neg_set_mark_test, pos_set_mark_test, set_of_rules) # zwrócenie liczby False Poistive'ów i False Negative'ów

    print("\nWYNIKI:")
    print(f"Dokładność: {accuracy(FN, FP, len(test_dataset))}%")
    print(f"Czułość: {recall(len(pos_set_mark_test) - FN, FN)}%")
    print(f"Precyzja: {recall(len(pos_set_mark_test) - FN, FP)}%")

if __name__ == "__main__":
    main()



