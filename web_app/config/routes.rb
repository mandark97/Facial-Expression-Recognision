Rails.application.routes.draw do
  root to: "predictions#index"
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
  resources :predictions do
    member do
      get :upvote
      get :downvote
    end
  end
end
