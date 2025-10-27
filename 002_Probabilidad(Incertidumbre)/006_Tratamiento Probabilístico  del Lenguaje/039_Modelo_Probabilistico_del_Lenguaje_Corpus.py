from collections import defaultdict
import numpy as np

class BigramLanguageModel:
    
    def __init__(self, smoothing_k=1):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.vocabulary = set()
        self.vocabulary_size = 0
        self.k = smoothing_k  # 'k' para el suavizado de Laplace (Add-k)

    def fit(self, corpus):
        """
        Entrenar el modelo (contar unigramas y bigramas)
        a partir de un corpus.
        """
        print("Iniciando entrenamiento (conteo de n-gramas)...")
        
        for sentence in corpus:
            # <s> = inicio de frase, </s> = fin de frase
            words = ["<s>"] + sentence.lower().split() + ["</s>"]
            
            # Actualizar el vocabulario
            self.vocabulary.update(words)
            
            # Contar unigramas (para los denominadores)
            for word in words:
                self.unigram_counts[word] += 1
                
            # Contar bigramas (para los numeradores)
            for i in range(len(words) - 1):
                prev_word = words[i]
                curr_word = words[i+1]
                self.bigram_counts[(prev_word, curr_word)] += 1
        
        self.vocabulary_size = len(self.vocabulary)
        
        print(f"Entrenamiento finalizado.")
        print(f"Tamaño del vocabulario (V): {self.vocabulary_size}")
        print("--- Algunos conteos de bigramas ---")
        print(f"('me', 'gusta'): {self.bigram_counts[('me', 'gusta')]}")
        print(f"('el', 'helado'): {self.bigram_counts[('el', 'helado')]}")
        print(f"('los', 'perros'): {self.bigram_counts[('los', 'perros')]}")

    def get_bigram_probability(self, prev_word, curr_word):
        """
        Calcula P(curr_word | prev_word) usando Suavizado de Laplace.
        
        Fórmula:
        P = (conteo(prev, curr) + k) / (conteo(prev) + k * V)
        V = tamaño del vocabulario
        k = factor de suavizado (usamos 1)
        """
        
        # Obtenemos los conteos (default 0 si no existen)
        bigram_count = self.bigram_counts.get((prev_word, curr_word), 0)
        unigram_count = self.unigram_counts.get(prev_word, 0)
        
        # Aplicamos Suavizado de Laplace (Add-1)
        numerator = bigram_count + self.k
        denominator = unigram_count + (self.k * self.vocabulary_size)
        
        return numerator / denominator

    def calculate_sentence_probability(self, sentence):
        """
        Calcula la probabilidad total de una frase
        multiplicando las probabilidades de sus bigramas.
        """
        
        words = ["<s>"] + sentence.lower().split() + ["</s>"]
        total_prob = 1.0
        
        print(f"\nCalculando probabilidad para: '{sentence}'")
        print("-" * 45)
        
        for i in range(len(words) - 1):
            prev_word = words[i]
            curr_word = words[i+1]
            
            # Obtener la probabilidad del bigrama
            prob = self.get_bigram_probability(prev_word, curr_word)
            
            print(f"  P({curr_word: <10} | {prev_word: <8}) = {prob:.4f}")
            
            # Multiplicamos la probabilidad
            total_prob *= prob
            
        print("-" * 45)
        print(f"  Probabilidad Total (log-prob): {np.log(total_prob):.4f}")
        print(f"  Probabilidad Total (bruta): {total_prob:.12f}")
        return total_prob

# --- 1. Definición del Corpus ---
corpus = [
    "me gusta el helado",
    "me gusta el chocolate",
    "me gustan los perros",
    "odio a los gatos",
    "me encanta mi perro"
]

# --- 2. Entrenamiento del Modelo ---
# Creamos una instancia y la entrenamos con el corpus
lm = BigramLanguageModel(smoothing_k=1)
lm.fit(corpus)

# --- 3. Prueba del Modelo ---
# Probamos con frases que el modelo ha visto,
# frases que no ha visto (pero plausibles),
# y frases que son gramaticalmente incorrectas.

# Frase 1: Vista en el corpus (debe ser alta)
prob1 = lm.calculate_sentence_probability("me gusta el helado")

# Frase 2: No vista, pero plausible (debe ser media)
# Nota: el bigrama ('gusta', 'mi') nunca se ha visto.
prob2 = lm.calculate_sentence_probability("me gusta mi perro")

# Frase 3: No vista y agramatical (debe ser muy baja)
prob3 = lm.calculate_sentence_probability("gatos odio chocolate")