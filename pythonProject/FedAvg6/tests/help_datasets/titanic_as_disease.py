from library.templates.partitioned_table import PartitionedPandasTable
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class TitanicAsDisease(PartitionedPandasTable):
    def get_dataset(self) -> pd.DataFrame:
        df = sns.load_dataset('titanic')
        df = df[['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
        # Treatment: sex (female=1, male=0)
        df['Treatment'] = LabelEncoder().fit_transform(df['sex'])  # female=1, male=0
        # Outcome: survived
        df['Outcome'] = df['survived']
        return df

dataset = TitanicAsDisease()
print(dataset.get_dataset())