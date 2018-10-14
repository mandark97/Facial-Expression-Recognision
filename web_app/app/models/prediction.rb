class Prediction < ApplicationRecord
  mount_uploader :image, ImageUploader

  enum status: {
    in_progress: 0,
    error: 1,
    completed: 2,
  }
  def vote(param)
    if param == 'upvote'
      Prediction.update_counters(self.id, upvote: 1)
    end
    if param == 'downvote'
      Prediction.update_counters(self.id, downvote: 1)
    end
  end
end
