import sqlite3
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
import shutil

class Memory:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.init_db()
        self.init_training_db()
        self.init_faiss()
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS events
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
             type TEXT,
             summary TEXT,
             text TEXT,
             analysis TEXT,
             image_path TEXT,
             meta TEXT)
        ''')
        
        c.execute("PRAGMA table_info(events)")
        existing_columns = [column[1] for column in c.fetchall()]
        
        if 'analysis' not in existing_columns:
            c.execute("ALTER TABLE events ADD COLUMN analysis TEXT")
            print("✅ Добавлен столбец 'analysis' в таблицу events")
        
        conn.commit()
        conn.close()
        
    def init_training_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS training_data
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
             class_name TEXT,
             description TEXT,
             image_path TEXT,
             features BLOB,
             is_verified BOOLEAN DEFAULT FALSE)
        ''')
        
        conn.commit()
        conn.close()
        
    def init_faiss(self):
        self.faiss_index_path = "memory_faiss.index"
        self.training_faiss_path = "training_faiss.index"
        dimension = 384
        
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
            if os.path.exists("faiss_ids.npy"):
                self.index_ids = np.load("faiss_ids.npy").tolist()
            else:
                self.index_ids = []
        else:
            self.index = faiss.IndexFlatIP(dimension)
            self.index_ids = []
            
        if os.path.exists(self.training_faiss_path):
            self.training_index = faiss.read_index(self.training_faiss_path)
            if os.path.exists("training_faiss_ids.npy"):
                self.training_index_ids = np.load("training_faiss_ids.npy").tolist()
            else:
                self.training_index_ids = []
        else:
            self.training_index = faiss.IndexFlatIP(dimension)
            self.training_index_ids = []
            
    def add_event(self, type_, summary, text, image_path=None, meta=None, analysis=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        meta_str = json.dumps(meta) if meta else None
        
        c.execute('''
            INSERT INTO events (type, summary, text, analysis, image_path, meta)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (type_, summary, text, analysis, image_path, meta_str))
        
        event_id = c.lastrowid
        
        content_to_index = text
        if analysis:
            content_to_index += " " + analysis
            
        embedding = self.embedder.encode(content_to_index)
        embedding = embedding.astype('float32')
        embedding = embedding / np.linalg.norm(embedding)
        self.index.add(np.array([embedding]))
        self.index_ids.append(event_id)
        
        faiss.write_index(self.index, self.faiss_index_path)
        np.save("faiss_ids.npy", np.array(self.index_ids))
        
        conn.commit()
        conn.close()
        return event_id
        
    def update_event_analysis(self, event_id, analysis):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT text, summary FROM events WHERE id = ?', (event_id,))
        result = c.fetchone()
        
        if result:
            text, summary = result
            
            c.execute('UPDATE events SET analysis = ? WHERE id = ?', (analysis, event_id))
            
            if event_id in self.index_ids:
                idx = self.index_ids.index(event_id)
                
                content_to_index = text + " " + analysis
                new_embedding = self.embedder.encode(content_to_index)
                new_embedding = new_embedding.astype('float32')
                new_embedding = new_embedding / np.linalg.norm(new_embedding)
                
                self.index.reconstruct(idx, new_embedding)
                faiss.write_index(self.index, self.faiss_index_path)
        
        conn.commit()
        conn.close()
        return True
        
    def delete_event(self, event_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT image_path FROM events WHERE id = ?', (event_id,))
        result = c.fetchone()
        
        if result:
            image_path = result[0]
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
        
        c.execute('DELETE FROM events WHERE id = ?', (event_id,))
        
        if event_id in self.index_ids:
            idx = self.index_ids.index(event_id)
            new_index = faiss.IndexFlatIP(384)
            new_ids = []
            
            for i, eid in enumerate(self.index_ids):
                if eid != event_id:
                    vector = self.index.reconstruct(i)
                    new_index.add(np.array([vector]))
                    new_ids.append(eid)
            
            self.index = new_index
            self.index_ids = new_ids
            
            faiss.write_index(self.index, self.faiss_index_path)
            np.save("faiss_ids.npy", np.array(self.index_ids))
        
        conn.commit()
        conn.close()
        return True
        
    def recent(self, limit=2500):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT id, timestamp, type, summary, text, analysis, image_path, meta 
            FROM events ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
        
    def search(self, query, k=5):
        if len(self.index_ids) == 0:
            return []
            
        query_embedding = self.embedder.encode(query)
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for i, score in zip(indices[0], scores[0]):
            if i < len(self.index_ids):
                event_id = self.index_ids[i]
                c.execute('SELECT id, timestamp, type, summary, text, analysis, image_path, meta FROM events WHERE id = ?', (event_id,))
                row = c.fetchone()
                if row:
                    results.append(row)
                    
        conn.close()
        return results
    
    def add_training_sample(self, class_name, description, image_path, features=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        features_blob = None
        if features is not None:
            features_blob = features.tobytes()
        
        c.execute('''
            INSERT INTO training_data (class_name, description, image_path, features)
            VALUES (?, ?, ?, ?)
        ''', (class_name, description, image_path, features_blob))
        
        sample_id = c.lastrowid
        
        if description:
            embedding = self.embedder.encode(description)
            embedding = embedding.astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            self.training_index.add(np.array([embedding]))
            self.training_index_ids.append(sample_id)
            
            faiss.write_index(self.training_index, self.training_faiss_path)
            np.save("training_faiss_ids.npy", np.array(self.training_index_ids))
        
        conn.commit()
        conn.close()
        return sample_id
    
    def get_training_samples(self, class_name=None, limit=1000):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if class_name:
            c.execute('''
                SELECT id, timestamp, class_name, description, image_path, is_verified
                FROM training_data 
                WHERE class_name = ? 
                ORDER BY timestamp DESC LIMIT ?
            ''', (class_name, limit))
        else:
            c.execute('''
                SELECT id, timestamp, class_name, description, image_path, is_verified
                FROM training_data 
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            
        rows = c.fetchall()
        conn.close()
        return rows
    
    def delete_training_sample(self, sample_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT image_path FROM training_data WHERE id = ?', (sample_id,))
        result = c.fetchone()
        
        if result:
            image_path = result[0]
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
        
        c.execute('DELETE FROM training_data WHERE id = ?', (sample_id,))
        
        if sample_id in self.training_index_ids:
            idx = self.training_index_ids.index(sample_id)
            new_index = faiss.IndexFlatIP(384)
            new_ids = []
            
            for i, sid in enumerate(self.training_index_ids):
                if sid != sample_id:
                    vector = self.training_index.reconstruct(i)
                    new_index.add(np.array([vector]))
                    new_ids.append(sid)
            
            self.training_index = new_index
            self.training_index_ids = new_ids
            
            faiss.write_index(self.training_index, self.training_faiss_path)
            np.save("training_faiss_ids.npy", np.array(self.training_index_ids))
        
        conn.commit()
        conn.close()
        return True
    
    def get_training_classes(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT DISTINCT class_name FROM training_data')
        classes = [row[0] for row in c.fetchall()]
        conn.close()
        return classes
    
    def verify_training_sample(self, sample_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE training_data SET is_verified = TRUE WHERE id = ?', (sample_id,))
        conn.commit()
        conn.close()
        return True