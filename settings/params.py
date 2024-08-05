"""Project parameters"""

MODEL_PARAMS = {
    "MIN_COMPLETION_RATE": 0.75,
    "TARGET_NAME": "SiteEnergyUse(kBtu)",
    "LOG_TRANSFORM": ['LotFrontage', 'LotArea', 'GarageArea', 'GrLivArea', '1stFlrSF', 'BsmtUnfSF', 'TotalBsmtSF'],
    'DEFAULT_FEATURE_NAMES': ['Alley','BsmtQual','ExterQual','Foundation','FullBath','GarageArea','GarageCars','GarageFinish','GarageType','GrLivArea','KitchenQualMSSubClass','Neighborhood','OverallQual','TotRmsAbvGrd','building_age','remodel_age','garage_age'],
    'TEST_SIZE': 0.20,
    "MIN_PPS": 0.05,
    "SEED": 42
}

SEED = 42