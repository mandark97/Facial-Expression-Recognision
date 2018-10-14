class AddStatusToPredictions < ActiveRecord::Migration[5.1]
  def change
    add_column :predictions, :status, :integer
  end
end
