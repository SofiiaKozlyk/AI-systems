import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv('renfe_small.csv')

    # Видалення пропущених значень
    df = df.dropna(subset=['price'])

    # Факторизація категорій
    df_encoded = pd.get_dummies(df[['origin', 'destination', 'train_type', 'train_class', 'fare']], drop_first=True)
    y = df['price']

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='price', bins=30)
    plt.title('Розподіл цін на квитки')
    plt.xlabel('Ціна (€)')
    plt.ylabel('Частота')

    plt.subplot(1, 3, 2)
    df['train_type'].value_counts().plot(kind='bar')
    plt.title('Розподіл за типом потяга (кількість)')
    plt.xlabel('Тип потяга')
    plt.ylabel('Кількість')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='train_type', y='price')
    plt.title('Розподіл за типом потяга (ціна)')
    plt.xlabel('Тип потяга')
    plt.ylabel('Ціна (€)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # байєсівська модель
    with pm.Model() as model:
        X_data = pm.Data("X_data", df_encoded.values.astype(float))
        y_data = y.astype(float)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=df_encoded.shape[1])
        intercept = pm.Normal("intercept", mu=30, sigma=20)
        mu = intercept + pm.math.dot(df_encoded.values, beta)
        sigma = pm.HalfNormal("sigma", sigma=10)
        likelihood = pm.Normal("price", mu=mu, sigma=sigma, observed=y_data)

        # семплінг
        trace = pm.sample(200, tune=100, target_accept=0.95)

    az.plot_trace(trace, var_names=["intercept", "sigma"], kind="trace")
    plt.tight_layout()
    plt.show()

    print(az.summary(trace))

    # передбачення ціни квитка з наступними даними
    ticket_data = {
        "origin": "MADRID",
        "destination": "BARCELONA",
        "train_type": "AVE",
        "train_class": "Turista",
        "fare": "Promo"
    }

    # створюємо новий рядок із нулями
    new_ticket = pd.DataFrame(0, index=[0], columns=df_encoded.columns)

    # встановлюємо 1 для потрібних ознак
    for col_prefix, val in zip(
        ["origin", "destination", "train_type", "train_class", "fare"],
        [ticket_data["origin"], ticket_data["destination"], ticket_data["train_type"], 
         ticket_data["train_class"], ticket_data["fare"]]
    ):
        col_name = f"{col_prefix}_{val}"
        if col_name in new_ticket.columns:
            new_ticket[col_name] = 1

    new_ticket_values = new_ticket.values.astype(float)

    print("\nПередбачення ціни для квитка:")
    print(ticket_data)

    # прогнозування ціни
    with model:
        pm.set_data({"X_data": new_ticket_values})
        posterior_predictive = pm.sample_posterior_predictive(trace)

    ppc_values = posterior_predictive["posterior_predictive"].price.values

    # середнє та стандартне відхилення прогнозу
    pred_mean = ppc_values.mean()
    pred_std = ppc_values.std()

    print(f"\nОчікувана ціна: {pred_mean} €, std: {pred_std} €")

if __name__ == "__main__":
    main()