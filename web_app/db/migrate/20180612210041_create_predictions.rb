class CreatePredictions < ActiveRecord::Migration[5.1]
  def change
    create_table :predictions do |t|
      t.string :image
      t.string :predicted_value

      t.timestamps
    end
  end
end
