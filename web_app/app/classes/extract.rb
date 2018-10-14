class Extract
  require 'xmlrpc/client'

  def get_prediction(prediction)
    result = server.call('extract', prediction.image.path)

    prediction.predictions = JSON.parse(result)
    prediction.status = :completed
    prediction.save!

  rescue => ex
    puts ex.message
    prediction.error!
  end

  def server
    @server ||= XMLRPC::Client.new3(host: 'localhost', port: 8000, timeout: 180)
  end
end
