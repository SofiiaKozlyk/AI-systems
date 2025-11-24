import pandas as pd

def calculate_probability(df, outlook, humidity, wind):
    total = len(df)
    p_yes = len(df[df['Play'] == 'Yes']) / total
    p_no = len(df[df['Play'] == 'No']) / total

    P_Outlook_Yes = len(df[(df['Outlook'] == outlook) & (df['Play'] == 'Yes')]) / len(df[df['Play'] == 'Yes'])
    P_Humidity_Yes = len(df[(df['Humidity'] == humidity) & (df['Play'] == 'Yes')]) / len(df[df['Play'] == 'Yes'])
    P_Wind_Yes = len(df[(df['Wind'] == wind) & (df['Play'] == 'Yes')]) / len(df[df['Play'] == 'Yes'])

    P_Outlook_No = len(df[(df['Outlook'] == outlook) & (df['Play'] == 'No')]) / len(df[df['Play'] == 'No'])
    P_Humidity_No = len(df[(df['Humidity'] == humidity) & (df['Play'] == 'No')]) / len(df[df['Play'] == 'No'])
    P_Wind_No = len(df[(df['Wind'] == wind) & (df['Play'] == 'No')]) / len(df[df['Play'] == 'No'])

    p_yes_score = P_Outlook_Yes * P_Humidity_Yes * P_Wind_Yes * p_yes
    p_no_score = P_Outlook_No * P_Humidity_No * P_Wind_No * p_no

    p_total = p_yes_score + p_no_score
    p_yes_final = p_yes_score / p_total
    p_no_final = p_no_score / p_total

    return p_yes_final, p_no_final


df = pd.read_csv('play_tennis.csv')
p_yes, p_no = calculate_probability(df, 'Sunny', 'High', 'Weak')
print(f"Ймовірність 'Yes': {p_yes:.4f}")
print(f"Ймовірність 'No': {p_no:.4f}")

if p_yes > p_no:
    print("\nМатч відбудеться.")
else:
    print("\nМатч не відбудеться.")
