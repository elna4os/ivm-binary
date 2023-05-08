### _Import Vector Machine_ for binary classification

---

- Scikit-learn compatible implementation
- Python used for dev: 3.8.16

---

To run tests:

```sh
PYTHONPATH=.:$PYTHONPATH pytest -rA -s
```

---

Breast cancer metrics (model default params)

```
ROCAUC = 0.972
            precision    recall  f1-score   support

        0       0.61      0.97      0.75        37
        1       0.98      0.63      0.77        63

accuracy                            0.76       100
macro avg       0.79      0.80      0.76       100
weighted avg    0.84      0.76      0.76       100
```
