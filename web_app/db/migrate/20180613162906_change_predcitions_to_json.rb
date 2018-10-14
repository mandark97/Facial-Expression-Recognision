class ChangePredcitionsToJson < ActiveRecord::Migration[5.1]
  def change
    remove_column :predictions, :predicted_value

    add_column :predictions, :predictions, :jsonb
  end
end
