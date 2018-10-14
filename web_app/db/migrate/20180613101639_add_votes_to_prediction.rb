class AddVotesToPrediction < ActiveRecord::Migration[5.1]
  def change
    add_column :predictions, :upvote, :integer, default: 0
    add_column :predictions, :downvote, :integer, default: 0
  end
end
