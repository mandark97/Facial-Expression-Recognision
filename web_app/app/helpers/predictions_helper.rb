module PredictionsHelper

  def emotions
    ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
  end

  def predicted_value(values)
    emotions[values.index(values.max)]
  end

  def face_img(image_path)
    image_path[0...-4] + '_face' + image_path[-4..-1]
  end
end
