class PredictionsController < ApplicationController

  def index
    @predictions = Prediction.all.order(id: :desc)
  end

  def show
    @prediction = Prediction.find(params[:id])
  end

  def new
    @prediction = Prediction.new

    respond_to do |format|
      format.html
      format.js
    end
  end

  def create
    @prediction = Prediction.new
    @prediction.image = prediction_params[:image]
    @prediction.status = :in_progress
    @prediction.save

    # result = ::Extract.new.get_prediction(@prediction)
    PredictionJob.perform_later(@prediction)
    redirect_to prediction_path(@prediction)
  end

  def destroy
    @prediction = Prediction.find(params[:id])

    @prediction.destroy

    redirect_to predictions_path
  end

  def upvote
    prediction = Prediction.find(params[:id])

    if cookies[prediction.id].nil?
      prediction.vote('upvote')
      prediction.save!
      cookies[prediction.id] = {
        value: true
      }
    else
    flash[:error] = 'Already voted'
    end
    redirect_to prediction_path(prediction)
  end

    def downvote
    prediction = Prediction.find(params[:id])

    if cookies[prediction.id].nil?
      prediction.vote('downvote')
      prediction.save!
      cookies[prediction.id] = {
        value: true
      }
    else
    flash[:error] = 'Already voted'
    end
    redirect_to prediction_path(prediction)
  end

  private

  def prediction_params
    params.require(:prediction).permit(:image)
  end


end
