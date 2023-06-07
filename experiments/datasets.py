import pandas as pd

def get_exams_dataset():
    df = pd.read_csv("../datasets/exams.csv")
    df = df.assign(score = lambda x: sum([df["math score"], df["reading score"], df["writing score"]])/3)
    bins = [0, 50, 60, 70, 80, 90, 100]
    category = ['2', '3', '3.5', '4', '4.5', '5']
    df['grade'] = pd.cut(df['score'], bins, labels=category)
    df.drop(columns=['math score', 'reading score','writing score', 'score'], inplace=True)

    for col in df:
        df[col] = df[col].astype(str)
        # df[col] = df[col].cat.codes # kodowanie atrubutów na inty
    feature_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']

    X = df[feature_cols] # Features
    y = df.grade # Target variable

    return X, y

def get_airline_dataset(set_type="train"):
    df = pd.read_csv(f"../datasets/airline_passenger_satisfaction_{set_type}.csv", index_col=0)

    df.drop(columns=['id'], inplace=True)

    age_bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    age_category = ['0-5', '6-15', '16-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-95']
    df['Age'] = pd.cut(df['Age'], age_bins, labels=age_category)

    flight_distance_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    flight_distance_category = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000', '3000-3500', '3500-4000', '4000-4500', '4500-5000']
    df['Flight Distance'] = pd.cut(df['Flight Distance'], flight_distance_bins, labels=flight_distance_category)

    for delay in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        delay_bins = [*range(0,int(max(df[delay])+60), 30)]
        delay_category = [f"{delay_bins[i]/60}h-{delay_bins[i+1]/60}h" for i in range(len(delay_bins)-1)]
        delay_category.insert(0, "0.0h")
        delay_bins.insert(0,-1)
        df[delay] = pd.cut(df[delay], delay_bins, labels=delay_category)

    for col in df:
        df[col] = df[col].astype(str)
        # df[col] = df[col].cat.codes # kodowanie atrubutów na inty
    feature_cols = ['Class', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']

    X = df.drop(columns=["satisfaction"]) # Features
    y = df.satisfaction # Target variable

    return X, y

def get_ecommerce_dataset():
    df = pd.read_csv(f"../datasets/e-commerce_shipping_data.csv") #, index_col=0
    df.drop(columns=['ID'], inplace=True)
    
    cost_of_product_bins = [*range(0,int(max(df["Cost_of_the_Product"])+30), 20)]
    cost_of_product_category = [f"{cost_of_product_bins[i]}-{cost_of_product_bins[i+1]}" for i in range(len(cost_of_product_bins)-1)]
    df["Cost_of_the_Product"] = pd.cut(df["Cost_of_the_Product"], cost_of_product_bins, labels=cost_of_product_category)

    discount_offered_bins = [*range(0,int(max(df["Discount_offered"])+20), 10)]
    discount_offered_category = [f"{discount_offered_bins[i]}-{discount_offered_bins[i+1]}" for i in range(len(discount_offered_bins)-1)]
    df["Discount_offered"] = pd.cut(df["Discount_offered"], discount_offered_bins, labels=discount_offered_category)

    weight_bins = [*range(int(min(df["Weight_in_gms"])-1),int(max(df["Weight_in_gms"])+20), 250)]
    weight_category = [f"{weight_bins[i]}-{weight_bins[i+1]}" for i in range(len(weight_bins)-1)]
    df["Weight_in_gms"] = pd.cut(df["Weight_in_gms"], weight_bins, labels=weight_category)

    for col in df:
        df[col] = df[col].astype(str)
        # df[col] = df[col].cat.codes # kodowanie atrubutów na inty

    X = df.drop(columns=["Reached.on.Time_Y.N"]) # Features
    y = df["Reached.on.Time_Y.N"] # Target variable

    return X, y
