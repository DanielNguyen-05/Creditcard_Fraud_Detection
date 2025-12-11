import numpy as np
import sys
import time

# --- HELPER: Hàm vẽ thanh tiến trình ---
def draw_progress_bar(current, total, metrics_str="", bar_length=30):
    """
    Vẽ thanh loading: [==============>.............] 50% | Loss: 0.1234
    """
    percent = float(current) * 100 / total
    arrow = '=' * int(percent / 100 * bar_length - 1) + '>'
    spaces = '.' * (bar_length - len(arrow))
    
    # Khi hoàn thành
    if current == total:
        arrow = '=' * bar_length
        spaces = ''

    sys.stdout.write(f"\r[{arrow}{spaces}] {int(percent)}% | {metrics_str}")
    sys.stdout.flush()

# ==========================================
# 1. LOGISTIC REGRESSION 
# ==========================================
class LogisticRegressionNumPy:
    def __init__(self, lr=1e-2, epochs=1000, batch_size=0, l2=0.0, verbose=True, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _loss(self, y_true, y_prob):
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        loss = - np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        if self.l2 > 0 and self.w is not None:
            loss += 0.5 * self.l2 * np.sum(self.w ** 2)
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = self.rng.normal(0, 0.01, size=(n_features,))
        self.b = 0.0
        batch_size = self.batch_size if (self.batch_size > 0) else n_samples
        
        if self.verbose:
            print(f"Training Logistic Regression ({self.epochs} epochs)...")

        for epoch in range(1, self.epochs + 1):
            perm = self.rng.permutation(n_samples)
            Xs, ys = X[perm], y[perm]
            
            for i in range(0, n_samples, batch_size):
                xb = Xs[i:i+batch_size]
                yb = ys[i:i+batch_size]
                z = xb.dot(self.w) + self.b
                preds = self._sigmoid(z)
                error = preds - yb
                gw = (xb.T.dot(error)) / xb.shape[0]
                gb = np.mean(error)
                if self.l2 > 0: gw += self.l2 * self.w
                self.w -= self.lr * gw
                self.b -= self.lr * gb
            
            # Tính loss để hiển thị
            current_prob = self._sigmoid(X.dot(self.w) + self.b)
            current_loss = self._loss(y, current_prob)
            self.loss_history.append(current_loss)

            if self.verbose:
                draw_progress_bar(epoch, self.epochs, f"Loss: {current_loss:.4f}")
        
        if self.verbose: print() 

    def predict_proba(self, X):
        return self._sigmoid(X.dot(self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ==========================================
# 2. LINEAR REGRESSION 
# ==========================================
class LinearRegressionNumPy:
    def __init__(self, lr=1e-3, epochs=2000, l2=0.0, verbose=True):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        if self.verbose:
            print(f"Training Linear Regression ({self.epochs} epochs)...")

        for epoch in range(1, self.epochs + 1):
            y_pred = X.dot(self.w) + self.b
            error = y_pred - y
            loss = np.mean(error ** 2)
            
            dw = (2 / n_samples) * X.T.dot(error)
            db = (2 / n_samples) * np.sum(error)
            
            if self.l2 > 0: dw += 2 * self.l2 * self.w
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            self.loss_history.append(loss)
            
            if self.verbose:
                draw_progress_bar(epoch, self.epochs, f"MSE: {loss:.4f}")
        
        if self.verbose: print()

    def predict_proba(self, X):
        return np.clip(X.dot(self.w) + self.b, 0, 1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ==========================================
# 3. GAUSSIAN NAIVE BAYES
# ==========================================
class GaussianNBNumPy:
    def __init__(self, verbose=True):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        self.verbose = verbose

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape

        if self.verbose:
            print(f"Training Naive Bayes on {n_classes} classes...")

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9 
            self.priors[idx] = X_c.shape[0] / n_samples
            
            if self.verbose:
                draw_progress_bar(idx + 1, n_classes, f"Class {c}")
        
        if self.verbose: print()

    def _pdf(self, class_idx, X):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (X - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            pdf_values = np.clip(self._pdf(idx, X), 1e-25, None)
            conditional = np.sum(np.log(pdf_values), axis=1)
            posteriors.append(prior + conditional)
        
        posteriors = np.array(posteriors).T 
        max_post = np.max(posteriors, axis=1, keepdims=True)
        exp_post = np.exp(posteriors - max_post)
        probs = exp_post / np.sum(exp_post, axis=1, keepdims=True)
        return probs[:, 1] 

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ==========================================
# 4. K-NEAREST NEIGHBORS
# ==========================================
class KNNNumPy:
    def __init__(self, k=5, verbose=True):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.verbose = verbose

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.verbose: 
            print(f"KNN Loaded: {X.shape[0]} samples")

    def predict(self, X_test):
        y_pred = []
        n_test = len(X_test)
        
        if self.verbose:
            print(f"KNN Predicting {n_test} samples...")

        for i, row in enumerate(X_test):
            # Calculate distance from this point to ALL training points
            # Euclidean distance
            distances = np.sqrt(np.sum((self.X_train - row)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
            
            if self.verbose and i % 10 == 0:
                draw_progress_bar(i + 1, n_test, f"Sample {i+1}")
        
        if self.verbose:
            draw_progress_bar(n_test, n_test, "Done!")
            print()
            
        return np.array(y_pred)

# --- METRICS ---
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1,1]; fp = cm[0,1]; fn = cm[1,0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def roc_curve(y_true, y_score):
    desc = np.argsort(-y_score)
    y_true_sorted = y_true[desc]
    y_score_sorted = y_score[desc]
    distinct_scores = np.unique(y_score_sorted)
    thresholds = np.r_[distinct_scores, distinct_scores[-1] - 1]
    if len(thresholds) > 2000:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, 2000))
    tprs, fprs = [], []
    P, N = np.sum(y_true == 1), np.sum(y_true == 0)
    for thr in thresholds:
        preds = (y_score >= thr).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tprs.append(tp / P if P > 0 else 0.0)
        fprs.append(fp / N if N > 0 else 0.0)
    return np.array(fprs), np.array(tprs), thresholds

def auc_from_roc(fprs, tprs):
    order = np.argsort(fprs)
    return np.trapz(tprs[order], fprs[order])