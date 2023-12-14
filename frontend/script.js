function predictWithSVM() {
    const audioInput = document.getElementById('audioFile');
    if (audioInput.files.length === 0) {
      alert('Please select a WAV file first.');
      return;
    }
    const audioFile = audioInput.files[0];
  
    // Convert the audio file to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioFile);
    reader.onload = function() {
      const base64Audio = reader.result.split(',')[1]; // Remove the content type prefix
  
      const data = { audio: base64Audio };
  
      // Replace 'backend_ip_address' and 'backend_port' with the actual values
      fetch('http://10.20.3.140:5000/predict_SVM', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      })
      .then(response => response.json())
      .then(data => {
        document.getElementById('predictionResult').textContent = `Predicted Genre: ${data.genre}`;
      })
      .catch(error => {
        console.error('Error:', error);
        document.getElementById('predictionResult').textContent = 'Error making prediction';
      });
    };
    reader.onerror = function(error) {
      console.log('Error: ', error);
    };
  }
  
  // Add event listener for file input to clear previous results
  document.getElementById('audioFile').addEventListener('change', () => {
    document.getElementById('predictionResult').textContent = '';
  });
  


  function predictWithVGG19() {
  const audioInput = document.getElementById('audioFile');
  if (audioInput.files.length === 0) {
  alert('Please select a WAV file first.');
  return;
  }
  const audioFile = audioInput.files[0];

  // Convert the audio file to base64
  const reader = new FileReader();
  reader.readAsDataURL(audioFile);
  reader.onload = function() {
  const base64Audio = reader.result.split(',')[1]; // Remove the content type prefix

  const data = { audio: base64Audio };

  // Replace 'vgg19_backend_container_ip' with the actual IP address of your VGG19 backend container
  // and 'vgg19_backend_port' with the port on which your backend server is listening.
  fetch('http://10.20.3.140:5001/predict_vgg', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('predictionResult').textContent = `Predicted Genre: ${data.genre}`;
  })
  .catch(error => {
    console.error('Error:', error);
    document.getElementById('predictionResult').textContent = 'Error making prediction with VGG19';
  });
  };
  reader.onerror = function(error) {
  console.log('Error: ', error);
  };
  }

  document.getElementById('audioFile').addEventListener('change', () => {
  document.getElementById('predictionResult').textContent = '';
  });

  // Add this line if you have a button with an onclick event for VGG19 prediction in your HTML
  document.querySelector('.vgg19-button').addEventListener('click', predictWithVGG19);

