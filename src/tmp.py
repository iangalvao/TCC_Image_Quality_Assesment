def print_experiment_info(experiments):
    for e in experiments:
        print(
            "- experiment_id: {}, name: {}, lifecycle_stage: {}".format(
                e.experiment_id, e.name, e.lifecycle_stage
            )
        )


predicted_actual = {"True values": y_val, "Predicted values": gridPredictionVal}
predicted_actual = pd.DataFrame(predicted_actual)

sns.scatterplot(
    data=predicted_actual,
    x="True values",
    y="Predicted values",
    color="black",
    alpha=0.5,
)
plt.axline((0, 0), slope=1, label="Perfect fit")
a_ = plt.title("Decision Tree Model\nPrediction vs Observed")
