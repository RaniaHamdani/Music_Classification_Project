pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Get the latest code from the source control
                git 'https://github.com/RaniaHamdani/Music_Classification_Project.git/'
            }
        }
        
        stage('Setup') {
            steps {
                // Set up a Python virtual environment and install dependencies
                sh '''
                virtualenv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }
        
        stage('Test') {
            steps {
                // Run your tests and generate a JUnit report
                sh '''
                . venv/bin/activate
                pytest --junitxml=reports/test-report.xml
                '''
            }
            
            post {
                // Publish the test results
                always {
                    junit 'reports/*.xml'
                }
            }
        }
    }
}
