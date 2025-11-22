from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

class TemplateLearner:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.templates = []
        
    def learn_templates(self, events, n_clusters=5):
        """Автоматическое обучение шаблонов отчетов из событий"""
        if len(events) < n_clusters:
            print("⚠️ Недостаточно событий для обучения шаблонов")
            return []
            
        texts = [event[4] for event in events if event[4]]  # text поле
        if not texts:
            return []
            
        # Векторизация текстов
        embeddings = self.embedder.encode(texts)
        
        # Кластеризация для выявления паттернов
        k = min(n_clusters, len(texts))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Создание шаблонов для каждого кластера
        templates = []
        for cluster_id in range(k):
            cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]
            
            if cluster_texts:
                # Анализ общих характеристик кластера
                template = self._analyze_cluster_pattern(cluster_texts, cluster_id)
                templates.append(template)
                
        self.templates = templates
        return templates
    
    def _analyze_cluster_pattern(self, texts, cluster_id):
        """Анализ паттернов в кластере текстов"""
        # Простой анализ ключевых слов и структуры
        words = []
        for text in texts:
            words.extend(text.lower().split())
        
        # Частотный анализ слов
        from collections import Counter
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(10) if len(word) > 3]
        
        # Анализ длины текстов
        avg_length = np.mean([len(text) for text in texts])
        
        return {
            "cluster_id": cluster_id,
            "sample_size": len(texts),
            "common_keywords": common_words,
            "average_length": avg_length,
            "example_template": self._generate_template_example(texts[0] if texts else "")
        }
    
    def _generate_template_example(self, text):
        """Генерация примера шаблона на основе текста"""
        # Простая замена конкретных значений на заполнители
        replacements = {
            r'\d{4}-\d{2}-\d{2}': '[ДАТА]',
            r'\d{2}:\d{2}:\d{2}': '[ВРЕМЯ]',
            r'[А-Яа-я]+\s+[А-Яа-я]+': '[ВИД_ОРГАНИЗМА]',
            r'\d+': '[КОЛИЧЕСТВО]'
        }
        
        import re
        template = text
        for pattern, replacement in replacements.items():
            template = re.sub(pattern, replacement, template)
            
        return template
    
    def get_template_suggestion(self, text):
        """Предложение шаблона для нового текста"""
        if not self.templates:
            return "Шаблоны еще не обучены"
            
        embedding = self.embedder.encode([text])
        
        # Находим ближайший кластер
        best_template = None
        best_similarity = -1
        
        for template in self.templates:
            # Простая эвристика на основе ключевых слов
            common_words = set(template['common_keywords'])
            text_words = set(text.lower().split())
            similarity = len(common_words.intersection(text_words)) / len(common_words.union(text_words))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_template = template
                
        return best_template if best_template else self.templates[0]