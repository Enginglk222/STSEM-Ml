import matplotlib.pyplot as plt
import shap
import os 

def explain_local(model, X_train, local_plots='local_plots', index = 0):
    # Create directories if they do not exist
    os.makedirs(local_plots, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    
    # Choosing an individual for local explanation
    observation = X_train.iloc[index]
    expected = model.predict([observation])[0]

    # Calculation SHAP values of selected individual(local observation) 
    shap_values = explainer.shap_values(observation)
    
    # Waterfall plot of local shapley values
    base_value = 0
    shap_values_matrix = shap.Explanation(values=shap_values, base_values=base_value, data=X_train, feature_names= X_train.columns)
    
    
    shap.waterfall_plot(shap_values_matrix[expected], max_display=len(X_train.columns), show = False )
    plt.tight_layout()

    plt.savefig(os.path.join(local_plots, f'local_graph_{index}.png'))
    plt.close()

    
def explain_global(model, X_train, global_plots='global_plots'):
    # Create directories if they do not exist
    os.makedirs(global_plots, exist_ok=True)

    explainer = shap.TreeExplainer(model)

    # Calculation SHAP values of whole dataset (global explanation)
    all_shap_values = explainer.shap_values(X_train)

    #Summary_plot of global shapley values
    shap.summary_plot(all_shap_values[0], features=X_train, show = False)

    plt.tight_layout()
    plt.savefig(os.path.join(global_plots, 'global.png'))
    plt.close()
