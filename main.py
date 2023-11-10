"""
This module imports utilities from utils.py. 

Google format guide followed

Originally written by : Rob
Refactored by : Roan 

Note:
    Any additional notes or considerations can be included in the 'Note' section.

"""

# sourcery skip: dont-import-test-modules
from utils import test_train_split

# global constants
SRC_PATH = "Data_In/nine_systems_data.csv"
TRAIN_PATH = "df_train.pickle"
TEST_PATH = "df_test.pickle"
VALIDATE_PATH = "df_validate.pickle"


def main():
    """Main.py code"""
    test_train_split(
        src_path=SRC_PATH,
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        validate_path=VALIDATE_PATH
    )


if __name__ == "__main__":
    main()
