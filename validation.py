import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load all the data files
annual_max = pd.read_csv('./results/annual_max_intensity.csv')
idf_gumbel = pd.read_csv('./results/idf_data.csv')
idf_svm = pd.read_csv('./results/idf_curves_SVM.csv')
idf_ann = pd.read_csv('./results/idf_curves_ANN.csv')
idf_tcn = pd.read_csv('./results/idf_curves_TCN.csv')
idf_tcan = pd.read_csv('./results/idf_curves_TCAN.csv')

print("Data loaded successfully!")
print(f"Annual max data shape: {annual_max.shape}")
print(f"Gumbel IDF data shape: {idf_gumbel.shape}")
print(f"SVM IDF data shape: {idf_svm.shape}")
print(f"ANN IDF data shape: {idf_ann.shape}")
print(f"TCN IDF data shape: {idf_tcn.shape}")
print(f"TCAN IDF data shape: {idf_tcan.shape}")

# Display column names to understand the structure
print("\nAnnual max columns:", annual_max.columns.tolist())
print("Gumbel IDF columns:", idf_gumbel.columns.tolist())
print("SVM IDF columns:", idf_svm.columns.tolist())

# Define the target durations and return periods
target_durations = ['10mns', '15mns', '90min', '2h', '12h', '15h', '18h', '24h']
target_return_periods = [2, 5, 10, 25, 50, 100]

# Map durations to minutes for AI model data
duration_mapping = {
    '5mns': 5,
    '10mns': 10,
    '15mns': 15,
    '30mns': 30,
    '1h': 60,
    '90min': 90,
    '2h': 120,
    '3h': 180,
    '6h': 360,
    '12h': 720,
    '15h': 900,
    '18h': 1080,
    '24h': 1440
}

# Map return periods to column names for AI models
return_period_mapping = {
    2: '2-year',
    5: '5-year',
    10: '10-year',
    25: '25-year'
}

# Map durations to Gumbel column names
gumbel_duration_mapping = {
    '5mns': '5 mins',
    '10mns': '10 mins',
    '15mns': '15 mins',
    '30mns': '30 mins',
    '1h': '60 mins',
    '90min': '90 mins',
    '2h': '120 mins',
    '3h': '180 mins',
    '6h': '360 mins',
    '12h': '720 mins',
    '15h': '900 mins',
    '18h': '1080 mins',
    '24h': '1440 mins'
}

print("\n" + "="*80)
print("RAINFALL INTENSITY PREDICTION ANALYSIS")
print("="*80)

# Function to get historical values for specific years back
def get_historical_values(years_back, duration):
    current_year = 2025
    target_year = current_year - years_back

    # Find the closest year in our data
    available_years = annual_max['year'].values
    closest_year = min(available_years, key=lambda x: abs(x - target_year))

    row = annual_max[annual_max['year'] == closest_year]
    if not row.empty:
        return row[duration].iloc[0], closest_year
    return None, None

# Function to get predicted values from each model
def get_predicted_values(duration_key, duration_minutes, return_period):
    results = {}

    # Get Gumbel value
    gumbel_row = idf_gumbel[idf_gumbel['Return Period (years)'] == return_period]
    if not gumbel_row.empty:
        gumbel_col = gumbel_duration_mapping.get(duration_key)
        if gumbel_col and gumbel_col in idf_gumbel.columns:
            results['Gumbel'] = gumbel_row[gumbel_col].iloc[0]

    # Get AI model values (only if duration is available in AI model data)
    if duration_minutes in [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]:
        models = {'SVM': idf_svm, 'ANN': idf_ann, 'TCN': idf_tcn, 'TCAN': idf_tcan}
        return_col = return_period_mapping[return_period]

        for model_name, model_data in models.items():
            model_row = model_data[model_data['Duration (minutes)'] == duration_minutes]
            if not model_row.empty:
                results[model_name] = model_row[return_col].iloc[0]

    return results

# Analyze for each target duration and return period
analysis_results = []

target_scenarios = [
    ('10mns', 2),
    ('5mns', 5),
    ('15mns', 25),
    ('30mns', 2),
    ('1h', 5),
    ('90min', 2),
    ('3h', 2),
    ('12h', 25),
    ('15h', 25),
    ('18h', 25),
    ('24h', 5),
    ('24h', 25),
]

for duration, return_period in target_scenarios:
    duration_minutes = duration_mapping.get(duration)
    if duration_minutes is None:
        continue

    print(f"\n{'='*20} ANALYSIS FOR {duration.upper()} DURATION, {return_period}-YEAR RETURN PERIOD {'='*20}")

    years_back = return_period
    historical_value, actual_year = get_historical_values(years_back, duration)

    if historical_value is None:
        continue

    predicted_values = get_predicted_values(duration, duration_minutes, return_period)

    if not predicted_values:  # Skip if no predictions available
        print(f"Historical Observed Value: {historical_value:.2f} mm/hr")
        print("No model predictions available for this duration")
        continue

    print(f"Historical Observed Value: {historical_value:.2f} mm/hr")

    # Calculate errors for each model
    errors = {}
    for model, predicted in predicted_values.items():
        error = abs(predicted - historical_value)
        relative_error = (error / historical_value) * 100
        errors[model] = {
            'predicted': predicted,
            'absolute_error': error,
            'relative_error': relative_error
        }
        print(f"{model:8}: {predicted:8.2f} mm/hr (Error: {error:6.2f}, {relative_error:5.1f}%)")

    # Find best performing model (lowest absolute error)
    best_model = min(errors.keys(), key=lambda x: errors[x]['absolute_error'])
    best_error = errors[best_model]['absolute_error']

    print(f"BEST MODEL: {best_model} (Absolute Error: {best_error:.2f} mm/hr)")

    # Calculate improvement over Gumbel
    if 'Gumbel' in errors and best_model != 'Gumbel':
        gumbel_error = errors['Gumbel']['absolute_error']
        improvement = ((gumbel_error - best_error) / gumbel_error) * 100
        print(f"IMPROVEMENT OVER GUMBEL: {improvement:.1f}%")

    analysis_results.append({
        'duration': duration,
        'return_period': return_period,
        'historical_value': historical_value,
        'actual_year': actual_year,
        'best_model': best_model,
        'best_error': best_error,
        'errors': errors
    })

print("\n" + "="*80)
print("SUMMARY RESULTS")
print("="*80)

# Summary by scenario
print("RESULTS BY SCENARIO:")
model_wins = {}
total_improvements = {}

for result in analysis_results:
    best_model = result['best_model']
    model_wins[best_model] = model_wins.get(best_model, 0) + 1

    # Calculate improvement over Gumbel
    if 'Gumbel' in result['errors'] and best_model != 'Gumbel':
        gumbel_error = result['errors']['Gumbel']['absolute_error']
        best_error = result['best_error']
        improvement = ((gumbel_error - best_error) / gumbel_error) * 100
        total_improvements[best_model] = total_improvements.get(best_model, []) + [improvement]

    duration = result['duration']
    rp = result['return_period']
    print(f"  {duration}-{rp}yr: {best_model} wins")

print(f"\nModel Performance (wins out of {len(analysis_results)} scenarios):")
for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
    avg_improvement = np.mean(total_improvements.get(model, [0])) if model in total_improvements else 0
    print(f"  {model:8}: {wins} wins, Average improvement over Gumbel: {avg_improvement:.1f}%")

print("\nOverall Best Performing Models:")
all_model_wins = {}
for result in analysis_results:
    model = result['best_model']
    all_model_wins[model] = all_model_wins.get(model, 0) + 1

for model, wins in sorted(all_model_wins.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: {wins} out of {len(analysis_results)} cases")

# Create a comprehensive summary table
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS TABLE")
print("="*80)

# Create a summary DataFrame for better visualization
summary_data = []
for result in analysis_results:
    duration = result['duration']
    return_period = result['return_period']
    best_model = result['best_model']
    best_error = result['best_error']
    historical_value = result['historical_value']

    # Calculate improvement over Gumbel if applicable
    improvement = 0
    if 'Gumbel' in result['errors'] and best_model != 'Gumbel':
        gumbel_error = result['errors']['Gumbel']['absolute_error']
        improvement = ((gumbel_error - best_error) / gumbel_error) * 100

    summary_data.append({
        'Duration': duration,
        'Return_Period': f"{return_period}-year",
        'Historical_Value': f"{historical_value:.2f}",
        'Best_Model': best_model,
        'Best_Error': f"{best_error:.2f}",
        'Improvement_over_Gumbel': f"{improvement:.1f}%" if improvement > 0 else "N/A"
    })

# Print the summary table
print(f"{'Duration':<8} {'Return':<8} {'Historical':<12} {'Best':<8} {'Error':<8} {'Improvement'}")
print(f"{'':8} {'Period':<8} {'Value (mm/hr)':<12} {'Model':<8} {'(mm/hr)':<8} {'over Gumbel'}")
print("-" * 80)

for data in summary_data:
    print(f"{data['Duration']:<8} {data['Return_Period']:<8} {data['Historical_Value']:<12} "
          f"{data['Best_Model']:<8} {data['Best_Error']:<8} {data['Improvement_over_Gumbel']}")

# Calculate overall statistics
total_cases = len(analysis_results)
ai_wins = sum(1 for result in analysis_results if result['best_model'] != 'Gumbel')
gumbel_wins = total_cases - ai_wins

print(f"\n{'='*80}")
print("FINAL STATISTICS")
print("="*80)
print(f"Total cases analyzed: {total_cases}")
print(f"AI models won: {ai_wins} cases ({ai_wins/total_cases*100:.1f}%)")
print(f"Gumbel won: {gumbel_wins} cases ({gumbel_wins/total_cases*100:.1f}%)")

# Calculate average improvement for AI models
ai_improvements = []
for result in analysis_results:
    if 'Gumbel' in result['errors'] and result['best_model'] != 'Gumbel':
        gumbel_error = result['errors']['Gumbel']['absolute_error']
        best_error = result['best_error']
        improvement = ((gumbel_error - best_error) / gumbel_error) * 100
        ai_improvements.append(improvement)

if ai_improvements:
    avg_improvement = np.mean(ai_improvements)
    print(f"Average improvement of AI models over Gumbel: {avg_improvement:.1f}%")

# ==============================================================================
# SAVE RESULTS TO CSV FILES
# ==============================================================================

print("\n" + "="*80)
print("SAVING RESULTS TO CSV FILES...")
print("="*80)

# Save detailed results for each scenario
detailed_results = []
for result in analysis_results:
    base_data = {
        'Duration': result['duration'],
        'Return_Period': result['return_period'],
        'Historical_Value': result['historical_value'],
        'Actual_Year': result['actual_year'],
        'Best_Model': result['best_model'],
        'Best_Error': result['best_error']
    }
    
    # Add predictions and errors for each model
    for model, error_data in result['errors'].items():
        base_data[f'{model}_Predicted'] = error_data['predicted']
        base_data[f'{model}_Absolute_Error'] = error_data['absolute_error']
        base_data[f'{model}_Relative_Error'] = error_data['relative_error']
    
    # Calculate improvement over Gumbel if applicable
    improvement = 0
    if 'Gumbel' in result['errors'] and result['best_model'] != 'Gumbel':
        gumbel_error = result['errors']['Gumbel']['absolute_error']
        improvement = ((gumbel_error - result['best_error']) / gumbel_error) * 100
    
    base_data['Improvement_over_Gumbel'] = improvement
    detailed_results.append(base_data)

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv('results/historical_validation_detailed.csv', index=False)
print("✅ Detailed validation results saved to: results/historical_validation_detailed.csv")

# Save summary statistics
summary_stats = {
    'Total_Cases': [total_cases],
    'AI_Wins': [ai_wins],
    'AI_Win_Percentage': [ai_wins/total_cases*100],
    'Gumbel_Wins': [gumbel_wins],
    'Gumbel_Win_Percentage': [gumbel_wins/total_cases*100],
    'Average_AI_Improvement': [avg_improvement if ai_improvements else 0]
}

# Add individual model performance
for model in ['Gumbel', 'SVM', 'ANN', 'TCN', 'TCAN']:
    wins = sum(1 for result in analysis_results if result['best_model'] == model)
    summary_stats[f'{model}_Wins'] = [wins]
    summary_stats[f'{model}_Win_Percentage'] = [wins/total_cases*100]
    
    # Average improvement for AI models
    if model != 'Gumbel' and model in total_improvements:
        avg_model_improvement = np.mean(total_improvements[model])
        summary_stats[f'{model}_Avg_Improvement'] = [avg_model_improvement]
    else:
        summary_stats[f'{model}_Avg_Improvement'] = [0]

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('results/historical_validation_summary.csv', index=False)
print("✅ Summary validation statistics saved to: results/historical_validation_summary.csv")

# Save model performance comparison
model_performance = []
for model, wins in sorted(all_model_wins.items(), key=lambda x: x[1], reverse=True):
    avg_improvement = np.mean(total_improvements.get(model, [0])) if model in total_improvements else 0
    model_performance.append({
        'Model': model,
        'Wins': wins,
        'Win_Percentage': wins/len(analysis_results)*100,
        'Average_Improvement_over_Gumbel': avg_improvement
    })

performance_df = pd.DataFrame(model_performance)
performance_df.to_csv('results/model_performance_comparison.csv', index=False)
print("✅ Model performance comparison saved to: results/model_performance_comparison.csv")

# ==============================================================================
# VISUALIZATION SECTION
# ==============================================================================

def create_validation_visualization(analysis_results):
    """Create visualization for validation results"""

    # Set up the figure with 6 subplots (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))


    # Use consistent color scheme matching evaluation.py
    model_colors = {'Gumbel': 'blue', 'SVM': 'green', 'ANN': 'red', 'TCN': 'orange', 'TCAN': 'purple'}

    # 1. AI vs Traditional Success Rate (Bar Chart instead of pie)
    ax1 = axes[0, 0]
    total_cases = len(analysis_results)
    ai_wins = sum(1 for result in analysis_results if result['best_model'] != 'Gumbel')
    gumbel_wins = total_cases - ai_wins

    categories = ['AI Models', 'Traditional\n(Gumbel)']
    wins = [ai_wins, gumbel_wins]
    colors = ['green', 'blue']  # Consistent with evaluation.py color scheme

    bars = ax1.bar(categories, wins, color=colors, alpha=0.8)
    ax1.set_title('AI Models vs Traditional Method\nSuccess Rate', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Best Predictions')

    # Add percentage labels
    for bar, win in zip(bars, wins):
        height = bar.get_height()
        percentage = (win/total_cases)*100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{win}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

    # 2. AI Model Performance Ranking (excluding Gumbel)
    ax2 = axes[0, 1]

    ai_wins_count = {}
    for result in analysis_results:
        if result['best_model'] != 'Gumbel':
            model = result['best_model']
            ai_wins_count[model] = ai_wins_count.get(model, 0) + 1

    if ai_wins_count:
        models = list(ai_wins_count.keys())
        wins = list(ai_wins_count.values())
        colors = [model_colors.get(model, 'gray') for model in models]

        bars = ax2.bar(models, wins, color=colors, alpha=0.8)
        ax2.set_title('AI Model Performance Ranking', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Number of Best Predictions')

        # Add value labels
        for bar, win in zip(bars, wins):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{win}', ha='center', va='bottom', fontweight='bold')

    # 3. Relative Error Comparison (AI vs Gumbel)
    ax3 = axes[0, 2]
    ai_errors = []
    gumbel_errors = []

    for result in analysis_results:
        if 'Gumbel' in result['errors']:
            gumbel_errors.append(result['errors']['Gumbel']['relative_error'])

            # Get the best AI model error for this case
            best_ai_error = float('inf')
            for model, error_data in result['errors'].items():
                if model != 'Gumbel' and error_data['relative_error'] < best_ai_error:
                    best_ai_error = error_data['relative_error']

            if best_ai_error != float('inf'):
                ai_errors.append(best_ai_error)

    data = [ai_errors, gumbel_errors]
    labels = ['Best AI Model', 'Traditional (Gumbel)']
    colors = ['green', 'blue']  # Consistent with evaluation.py color scheme

    box_plot = ax3.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_title('Prediction Error Comparison\n(Lower is Better)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Relative Error (%)')
    ax3.grid(True, alpha=0.3)

    # 4. Historical vs Predicted - AI Models Only
    ax4 = axes[1, 0]

    for model in ['SVM', 'ANN', 'TCN', 'TCAN']:  # Exclude Gumbel
        historical_vals = []
        predicted_vals = []

        for result in analysis_results:
            if model in result['errors']:
                historical_vals.append(result['historical_value'])
                predicted_vals.append(result['errors'][model]['predicted'])

        if historical_vals and predicted_vals:
            ax4.scatter(historical_vals, predicted_vals, label=model,
                       color=model_colors.get(model, 'gray'), alpha=0.7, s=60)

    # Perfect prediction line
    min_val = min([result['historical_value'] for result in analysis_results])
    max_val = max([result['historical_value'] for result in analysis_results])
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
             label='Perfect Prediction', linewidth=2)

    ax4.set_xlabel('Historical Observed (mm/hr)', fontweight='bold')
    ax4.set_ylabel('AI Model Predicted (mm/hr)', fontweight='bold')
    ax4.set_title('AI Model Accuracy vs Historical Data', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Improvement Magnitude over Traditional Method
    ax5 = axes[1, 1]

    # Calculate improvement for all AI wins
    all_improvements = []
    for result in analysis_results:
        if 'Gumbel' in result['errors'] and result['best_model'] != 'Gumbel':
            gumbel_error = result['errors']['Gumbel']['absolute_error']
            best_error = result['best_error']
            improvement = ((gumbel_error - best_error) / gumbel_error) * 100
            all_improvements.append(improvement)

    if all_improvements:
        # Create histogram of improvements
        ax5.hist(all_improvements, bins=6, color='orange', alpha=0.7, edgecolor='black')  # Use orange to distinguish from model-specific colors
        ax5.set_title('Distribution of AI Improvement\nover Traditional Method', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Improvement Percentage (%)')
        ax5.set_ylabel('Number of Cases')
        ax5.grid(True, alpha=0.3)

        # Add mean line
        mean_improvement = np.mean(all_improvements)
        ax5.axvline(mean_improvement, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_improvement:.1f}%')
        ax5.legend()

    # 6. AI Model Superiority by Scenario
    ax6 = axes[1, 2]

    # Calculate win rate for each AI model when they compete
    scenario_performance = {}
    for result in analysis_results:
        duration = result['duration']
        rp = result['return_period']
        scenario = f"{duration}\n{rp}yr"

        if result['best_model'] != 'Gumbel':
            scenario_performance[scenario] = result['best_model']

    # Create a visualization using individual model colors
    scenarios = list(scenario_performance.keys())
    ai_models = ['SVM', 'ANN', 'TCN', 'TCAN']

    if scenarios:
        # Create colored rectangles for each winning model
        for i, scenario in enumerate(scenarios):
            winning_model = scenario_performance[scenario]
            for j, model in enumerate(ai_models):
                if model == winning_model:
                    # Use the specific model color
                    color = model_colors.get(model, 'gray')
                    rect = plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8,
                                       facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
                    ax6.add_patch(rect)

                    # Add checkmark
                    ax6.text(j, i, "✓", ha="center", va="center",
                            color="white", fontweight='bold', fontsize=14)

        # Set proper axis limits and configuration
        ax6.set_xlim(-0.5, len(ai_models) - 0.5)
        ax6.set_ylim(-0.5, len(scenarios) - 0.5)
        ax6.set_xticks(range(len(ai_models)))
        ax6.set_xticklabels(ai_models)
        ax6.set_yticks(range(len(scenarios)))
        ax6.set_yticklabels(scenarios, fontsize=10)
        ax6.set_xlabel('AI Models', fontweight='bold')
        ax6.set_ylabel('Test Scenarios', fontweight='bold')
        ax6.set_title('AI Model Wins by Scenario\n(✓ = Best Performance)', fontweight='bold', fontsize=12)

        # Remove grid and set background
        ax6.set_facecolor('lightgray')
        ax6.grid(False)

    plt.tight_layout()
    plt.savefig('figures/historical_validation_analysis.png', dpi=300, bbox_inches='tight')
    
    # Save individual subplots
    import os
    subplot_dir = 'figures/validation_subplots'
    os.makedirs(subplot_dir, exist_ok=True)
    
    # Store variables needed for individual subplots
    # For subplot 1: AI vs Traditional
    subplot1_categories = ['AI Models', 'Traditional\n(Gumbel)']
    subplot1_wins = [ai_wins, gumbel_wins]
    subplot1_colors = ['green', 'blue']
    
    # For subplot 2: AI Model Performance
    subplot2_models = list(ai_wins_count.keys()) if ai_wins_count else []
    subplot2_wins = list(ai_wins_count.values()) if ai_wins_count else []
    subplot2_colors = [model_colors.get(model, 'gray') for model in subplot2_models]
    
    # For subplot 3: Error comparison data already available
    
    # Save each subplot individually
    subplot_titles = [
        'ai_vs_traditional_success_rate',
        'ai_model_performance_ranking', 
        'prediction_error_comparison',
        'ai_accuracy_vs_historical',
        'improvement_distribution',
        'ai_model_wins_by_scenario'
    ]
    
    for i, title in enumerate(subplot_titles):
        # Create a new figure for each subplot
        fig_sub = plt.figure(figsize=(8, 6))
        ax_sub = fig_sub.add_subplot(111)
        
        # Copy the content from the original subplot
        if i == 0:  # AI vs Traditional Success Rate
            bars = ax_sub.bar(subplot1_categories, subplot1_wins, color=subplot1_colors, alpha=0.8)
            ax_sub.set_title('AI Models vs Traditional Method\nSuccess Rate', fontweight='bold', fontsize=12)
            ax_sub.set_ylabel('Number of Best Predictions')
            for bar, win in zip(bars, subplot1_wins):
                height = bar.get_height()
                percentage = (win/total_cases)*100
                ax_sub.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{win}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        elif i == 1:  # AI Model Performance Ranking
            if subplot2_models:
                bars = ax_sub.bar(subplot2_models, subplot2_wins, color=subplot2_colors, alpha=0.8)
                ax_sub.set_title('AI Model Performance Ranking', fontweight='bold', fontsize=12)
                ax_sub.set_ylabel('Number of Best Predictions')
                for bar, win in zip(bars, subplot2_wins):
                    height = bar.get_height()
                    ax_sub.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               f'{win}', ha='center', va='bottom', fontweight='bold')
        
        elif i == 2:  # Relative Error Comparison
            box_plot = ax_sub.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax_sub.set_title('Prediction Error Comparison\n(Lower is Better)', fontweight='bold', fontsize=12)
            ax_sub.set_ylabel('Relative Error (%)')
            ax_sub.grid(True, alpha=0.3)
        
        elif i == 3:  # Historical vs Predicted
            for model in ['SVM', 'ANN', 'TCN', 'TCAN']:
                historical_vals = []
                predicted_vals = []
                for result in analysis_results:
                    if model in result['errors']:
                        historical_vals.append(result['historical_value'])
                        predicted_vals.append(result['errors'][model]['predicted'])
                if historical_vals and predicted_vals:
                    ax_sub.scatter(historical_vals, predicted_vals, label=model,
                                  color=model_colors.get(model, 'gray'), alpha=0.7, s=60)
            min_val = min([result['historical_value'] for result in analysis_results])
            max_val = max([result['historical_value'] for result in analysis_results])
            ax_sub.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
                       label='Perfect Prediction', linewidth=2)
            ax_sub.set_xlabel('Historical Observed (mm/hr)', fontweight='bold')
            ax_sub.set_ylabel('AI Model Predicted (mm/hr)', fontweight='bold')
            ax_sub.set_title('AI Model Accuracy vs Historical Data', fontweight='bold', fontsize=12)
            ax_sub.legend()
            ax_sub.grid(True, alpha=0.3)
        
        elif i == 4:  # Improvement Distribution
            if all_improvements:
                ax_sub.hist(all_improvements, bins=6, color='orange', alpha=0.7, edgecolor='black')
                ax_sub.set_title('Distribution of AI Improvement\nover Traditional Method', fontweight='bold', fontsize=12)
                ax_sub.set_xlabel('Improvement Percentage (%)')
                ax_sub.set_ylabel('Number of Cases')
                ax_sub.grid(True, alpha=0.3)
                mean_improvement = np.mean(all_improvements)
                ax_sub.axvline(mean_improvement, color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {mean_improvement:.1f}%')
                ax_sub.legend()
        
        elif i == 5:  # AI Model Wins by Scenario
            if scenarios:
                for j, scenario in enumerate(scenarios):
                    winning_model = scenario_performance[scenario]
                    for k, model in enumerate(ai_models):
                        if model == winning_model:
                            color = model_colors.get(model, 'gray')
                            rect = plt.Rectangle((k-0.4, j-0.4), 0.8, 0.8,
                                               facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
                            ax_sub.add_patch(rect)
                            ax_sub.text(k, j, "✓", ha="center", va="center",
                                       color="white", fontweight='bold', fontsize=14)
                ax_sub.set_xlim(-0.5, len(ai_models) - 0.5)
                ax_sub.set_ylim(-0.5, len(scenarios) - 0.5)
                ax_sub.set_xticks(range(len(ai_models)))
                ax_sub.set_xticklabels(ai_models)
                ax_sub.set_yticks(range(len(scenarios)))
                ax_sub.set_yticklabels(scenarios, fontsize=10)
                ax_sub.set_xlabel('AI Models', fontweight='bold')
                ax_sub.set_ylabel('Test Scenarios', fontweight='bold')
                ax_sub.set_title('AI Model Wins by Scenario\n(✓ = Best Performance)', fontweight='bold', fontsize=12)
                ax_sub.set_facecolor('lightgray')
                ax_sub.grid(False)
        
        # Save the individual subplot
        fig_sub.savefig(f'{subplot_dir}/{title}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_sub)
    
    plt.show()

# Create the visualization
print("\n" + "="*80)
print("CREATING VISUALIZATION...")
print("="*80)

create_validation_visualization(analysis_results)

print("✅ Visualization created and saved!")
print("📊 File saved: figures/historical_validation_analysis.png")
print("📊 Individual subplots saved in: figures/validation_subplots/")

print("\n" + "="*80)
print("ALL RESULTS SAVED SUCCESSFULLY!")
print("="*80)
print("📊 Files created:")
print("  - results/historical_validation_detailed.csv")
print("  - results/historical_validation_summary.csv") 
print("  - results/model_performance_comparison.csv")
print("  - figures/historical_validation_analysis.png")
print("  - figures/validation_subplots/ (6 individual subplot images)")
print("="*80)
