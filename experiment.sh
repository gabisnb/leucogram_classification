echo "Running 5 fold experiment"

# fold 1
echo "Fold 1"
python3 train_folds.py -i 1 -f 1

# fold 2
echo "Fold 2"
python3 train_folds.py -i 2 -f 2

# fold 3
echo "Fold 3"
python3 train_folds.py -i 3 -f 3

# fold 4
echo "Fold 4"
python3 train_folds.py -i 4 -f 4

# fold 5
echo "Fold 5"
python3 train_folds.py -i 5 -f 5