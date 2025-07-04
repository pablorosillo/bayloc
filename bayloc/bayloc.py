from collections import defaultdict, Counter
from datetime import timedelta, datetime
import numpy as np
from scipy.optimize import minimize_scalar
from .utils import daterange
from . import rezin  # assumes rezin.py is in the repo and installable
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class LambdaOptimizationError(Exception):
    pass

class BayLoc:
    def __init__(self, raw_data_set, lambda_min=0.0001, lambda_max=75):
        """
        raw_data_set: set of tuples like {('2015-06-13', 'FR'), ('2015-05-21', 'DE'), ...}
        lambda_min, lambda_max: bounds for lambda optimization
        """
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Convert raw data to daily_timeline: {date_obj: Counter({country: count, ...}), ...}
        self.daily_timeline = self._convert_input(raw_data_set)
        self.consolidated_timeline = None
        self.best_lambda = None
        self.guessed_timeline = None

    def _convert_input(self, raw_data_set):
        daily_timeline = defaultdict(Counter)
        for date_str, country in raw_data_set:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            daily_timeline[date_obj][country] += 1
        return dict(daily_timeline)

    def _consolidate_timeline(self):
        consolidated = {}
        for date, countries in self.daily_timeline.items():
            max_count = max(countries.values())
            max_countries = [c for c, count in countries.items() if count == max_count]
            consolidated[date] = max_countries[-1] if len(max_countries) > 1 else max_countries[0]
        self.consolidated_timeline = dict(sorted(consolidated.items(), key=lambda x: x[0]))

    def _optimize_lambda(self):
        known_days = list(self.consolidated_timeline.keys())

        def compute_likelihood(lambd):
            likelihood = 0
            for target_day in known_days:
                target_loc = self.consolidated_timeline[target_day]
                weights = defaultdict(float)
                total = 0.0
                for other_day in known_days:
                    if other_day == target_day:
                        continue
                    loc = self.consolidated_timeline[other_day]
                    dist = abs((target_day - other_day).days)
                    weight = 1 / ((dist + 1e-3) ** lambd)
                    weights[loc] += weight
                    total += weight
                prob = weights[target_loc] / (total + 1e-9)
                likelihood += np.log(prob + 1e-9)
            return -likelihood

        res = minimize_scalar(
            compute_likelihood,
            bounds=(self.lambda_min, self.lambda_max),
            method='bounded',
            options={'xatol': 1.0}
        )
        self.best_lambda = res.x
        if abs(self.best_lambda - self.lambda_max) < 1e-5:
            raise LambdaOptimizationError(
                f"Lambda optimization hit the upper bound ({self.lambda_max}). "
                "Consider increasing lambda_max."
            )

    def infer_missing_days(self):
        if self.consolidated_timeline is None:
            self._consolidate_timeline()

        start_date = min(self.consolidated_timeline.keys())
        end_date = max(self.consolidated_timeline.keys())
        all_days = list(daterange(start_date, end_date))
        missing_days = [day for day in all_days if day not in self.consolidated_timeline]

        self._optimize_lambda()

        guessed = {}
        for missing_day in missing_days:
            weights = defaultdict(float)
            for known_day, loc in self.consolidated_timeline.items():
                dist = abs((missing_day - known_day).days)
                weights[loc] += 1 / ((dist + 1e-3) ** self.best_lambda)
            total_weight = sum(weights.values())
            probabilities = {loc: w / total_weight for loc, w in weights.items()}
            guessed_location = str(np.random.choice(list(probabilities.keys()), p=list(probabilities.values())))
            guessed[missing_day] = guessed_location

        self.guessed_timeline = guessed

        full_timeline = dict(self.consolidated_timeline)
        full_timeline.update(guessed)
        self.full_timeline = dict(sorted(full_timeline.items(), key=lambda x: x[0]))
        return self.full_timeline

    def get_residence_history(self, n_days):
        """
        Returns the residence dictionary based on minimum residency days threshold.

        Requires infer_missing_days() has been run.
        """
        if self.full_timeline is None:
            self.infer_missing_days()

        location_days_list = []
        current_location = None
        current_streak = 0

        for location in self.full_timeline.values():
            if location == current_location:
                current_streak += 1
            else:
                if current_location is not None:
                    location_days_list.append((current_location, current_streak))
                current_location = location
                current_streak = 1
        if current_location is not None:
            location_days_list.append((current_location, current_streak))

        res_history = rezin.rezin(location_days_list, n_days)

        residence_dict = defaultdict(list)
        dates = list(self.full_timeline.keys())
        for i, (country, start_idx) in enumerate(res_history):
            start_date = dates[start_idx]
            if i + 1 < len(res_history):
                next_start_idx = res_history[i + 1][1]
                end_date = dates[next_start_idx - 1]
            else:
                end_date = dates[-1]
            residence_dict[str(country)].append([start_date.isoformat(), end_date.isoformat()])
        return residence_dict if residence_dict else None

    def get_basic_stats(self):
        """
        Returns statistics for both the original input and the inferred timeline.

        Includes:
        - total_days
        - unique_locations
        - counts_per_location
        - start_date and end_date
        - number of known and inferred days
        """
        from collections import defaultdict

        # Stats from original input (daily_timeline)
        input_locations = []
        input_counts = defaultdict(int)
        for countries in self.daily_timeline.values():
            for loc, count in countries.items():
                input_locations.append(loc)
                input_counts[loc] += count

        input_dates = list(self.daily_timeline.keys())
        input_stats = {
            "total_days": len(self.daily_timeline),
            "unique_locations": set(input_locations),
            "counts_per_location": dict(input_counts),
            "start_date": min(input_dates) if input_dates else None,
            "end_date": max(input_dates) if input_dates else None,
        }

        # Ensure inference has been run
        if self.full_timeline is None:
            self.infer_missing_days()

        # Stats from full timeline (after inference)
        inferred_locations = list(self.full_timeline.values())
        inferred_counts = defaultdict(int)
        for loc in inferred_locations:
            inferred_counts[loc] += 1

        inferred_dates = list(self.full_timeline.keys())
        known_dates = set(self.consolidated_timeline.keys())
        total_dates = set(inferred_dates)
        inferred_only_dates = total_dates - known_dates

        inferred_stats = {
            "total_days": len(self.full_timeline),
            "unique_locations": set(inferred_locations),
            "counts_per_location": dict(inferred_counts),
            "start_date": min(inferred_dates),
            "end_date": max(inferred_dates),
            "number_of_known_days": len(known_dates),
            "number_of_inferred_days": len(inferred_only_dates),
            "optimized_lambda": self.best_lambda if self.best_lambda is not None else None
        }

        return {
            "input_data": input_stats,
            "inferred_timeline": inferred_stats
        }


    def compute_auroc_training_subset(self, r):
        """
        Compute micro-average AUROC over a test subset, using a model trained on training subset of known timeline.

        Parameters:
        - r: float between 0 and 1 (exclusive), fraction of known days used for training.

        Returns:
        - float: micro-average AUROC score on test subset.
        """

        if not (0 < r < 1):
            raise ValueError("Parameter r must be between 0 and 1 (exclusive).")

        # Extract known timeline items (date, location), sorted by date
        known_items = sorted(self.consolidated_timeline.items(), key=lambda x: x[0])
        dates, locs = zip(*known_items)

        unique_classes = sorted(set(locs))
        n_samples = len(dates)
        n_train = max(int(n_samples * r), len(unique_classes))  # ensure all classes present in train

        # Split dates into training and test sets
        train_dates = dates[:n_train]
        train_locs = locs[:n_train]
        test_dates = dates[n_train:]
        test_locs = locs[n_train:]

        # Build consolidated timeline dict for training only
        train_timeline = dict(zip(train_dates, train_locs))

        # Lambda optimization on training data
        def compute_likelihood(lambd):
            likelihood = 0
            for target_day in train_dates:
                target_loc = train_timeline[target_day]
                weights = defaultdict(float)
                total = 0.0
                for other_day in train_dates:
                    if other_day == target_day:
                        continue
                    loc = train_timeline[other_day]
                    dist = abs((target_day - other_day).days)
                    weight = 1 / ((dist + 1e-3) ** lambd)
                    weights[loc] += weight
                    total += weight
                prob = weights[target_loc] / (total + 1e-9)
                likelihood += np.log(prob + 1e-9)
            return -likelihood

        res = minimize_scalar(
            compute_likelihood,
            bounds=(self.lambda_min, self.lambda_max),
            method='bounded',
            options={'xatol': 1.0}
        )
        best_lambda = res.x

        # Predict location probabilities for test dates
        y_true = []
        y_score = []

        lb = LabelBinarizer()
        lb.fit(unique_classes)  # fit on all classes to keep consistent columns

        for test_day, true_loc in zip(test_dates, test_locs):
            weights = defaultdict(float)
            for train_day, train_loc in train_timeline.items():
                dist = abs((test_day - train_day).days)
                weights[train_loc] += 1 / ((dist + 1e-3) ** best_lambda)
            total_weight = sum(weights.values())
            probs = np.array([weights.get(cls, 0) / total_weight for cls in unique_classes])

            y_score.append(probs)
            y_true.append(true_loc)

        y_score = np.array(y_score)
        y_true_bin = lb.transform(y_true)
        if y_true_bin.shape[1] == 1:  # binary case fix
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        # Compute micro-average ROC curve and AUC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        return roc_auc_micro



