class PredictionJob < ApplicationJob
  def perform(prediction)
    Extract.new.get_prediction(prediction)
  end
end