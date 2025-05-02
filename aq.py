import pandas as pd
import copy

df = pd.read_csv("dane_pogoda.csv")



def initialize_general_complex(df):
    general_rule = {}
    for attr in df.columns:
        if attr != 'class':  # pomijamy etykietę klasy
            general_rule[attr] = set(df[attr].unique())
    return [general_rule]  # zwracamy listę reguł (na start jedna)



def positive_examples(seed): # zwraca dataset przykładów klasy pozytywnej
    return df[df['class'] == seed]



def negative_examples(seed): # zwraca dataset przykładów klasy negatywnej
    return df[df['class'] != seed]



def return_positive_seed(pos_set):
    pos_set = list(pos_set.to_dict(orient='records'))
    return pos_set[0]



def return_negative_seed(neg_set):
    neg_set = list(neg_set.to_dict(orient='records'))
    return neg_set[0]



def covers(rule, example):

    for attr in rule:
        if example[attr] not in rule[attr]:
            return False
    return True



def check_coverage_of_negative_examples(neg_set, complexes):
    filtered_rows = []
    for _, example in neg_set.iterrows():
        if any(covers(rule, example) for rule in complexes):
            filtered_rows.append(example)

    return pd.DataFrame(filtered_rows)



def check_coverage_of_positive_examples(pos_set, complexes):
    filtered_rows = []
    for _, example in pos_set.iterrows():
        if not any(covers(rule, example) for rule in complexes):
            filtered_rows.append(example)

    return pd.DataFrame(filtered_rows)



def compare_seeds(pos_seed, neg_seed):
    differing_attrs = []
    for i, attr in enumerate(pos_seed):
        if attr == 'class':
            continue
        if pos_seed[attr] != neg_seed[attr]:
            differing_attrs.append(attr)  # index + name
    return differing_attrs



def generate_complex(compared_seeds, neg_seed, complex):
    final_complex = []

    for comp in range(len(complex)):    

        for attr in range(len(compared_seeds)):
            temp_complex = copy.deepcopy(complex)
            val = temp_complex[comp][compared_seeds[attr]]
            
            val.discard(neg_seed[compared_seeds[attr]])
            
            #temp_complex[comp].remove(compared_seeds[attr])
    #print(temp_complex)
            final_complex.append(temp_complex[comp])
    return final_complex



def evaluate_complexes_LEF(complexes, positive_examples, negative_examples, m=1):
    def f1(complex):
        # Liczba pokrywanych pozytywnych przykładów
        return sum(covers(complex, ex) for _, ex in positive_examples.iterrows())

    def f2(complex):
        # Liczba NIEpokrywanych negatywnych przykładów
        return sum(not covers(complex, ex) for _, ex in negative_examples.iterrows())

    # Nadaj ocenę każdemu kompleksowi w==
    scored = []
    for idx, c in enumerate(complexes):
        score = (f1(c))
        scored.append((score, idx, c))  # Dodaj indeks jako tie-breaker

    # Sortuj malejąco wg score, a przy remisie preferuj wyższy indeks (czyli późniejszy kompleks)
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))

    # Zwróć m najlepszych kompleksów
    best = [c for _, _, c in scored[:m]]
    return best



def main():
    
    seed = 0
    print(df['class'])
    neg_set_mark = negative_examples(seed)
    pos_set = positive_examples(seed)
    pos_set_mark = positive_examples(seed)
    set_of_rules = []

    while not pos_set.empty : # dopóki zbiór reguł nie pokrywa wszystkich przykładów
        
        complex = initialize_general_complex(df)
        pos_seed = return_positive_seed(pos_set)
        print('pos seed')
        print(pos_seed)
 
        neg_set = neg_set_mark
        while not neg_set.empty : # dopóki zbiór negatywny nie jest pusty
            neg_seed = return_negative_seed(neg_set)
            print('neg seed')
            print(neg_seed)
            compared_seeds = compare_seeds(pos_seed, neg_seed)

            complex = generate_complex(compared_seeds, neg_seed, complex)
            print('zestaw complexów')
            print(complex)
            neg_set = check_coverage_of_negative_examples(neg_set,complex)

            complex = evaluate_complexes_LEF(complex, pos_set, neg_set_mark)
            print('po ocenie')
            print(complex)


        print('dodanie zasady do set of rules')
        print(complex)
        set_of_rules.append(complex)
        pos_set = check_coverage_of_positive_examples(pos_set, complex)
        print('to zbiór pozytywnych przykładów')
        print(pos_set)
    
    for i in set_of_rules:
        print(i)

        

main()




