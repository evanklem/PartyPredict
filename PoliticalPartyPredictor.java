import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Political Party Predictor - A machine learning application that predicts
 * political party affiliation based on survey responses using Naive Bayes classification.
 * 
 * This application includes:
 * - Interactive console-based survey
 * - CSV-based data storage
 * - Naive Bayes classifier implementation
 * - Real-time prediction capabilities
 * - Model training and evaluation
 * - Weighted response system
 */
public class PoliticalPartyPredictor {
    
    // Political party categories
    public enum PoliticalParty {
        DEMOCRAT, REPUBLICAN, LIBERTARIAN, GREEN
    }
    
    // Survey question structure
    static class Question {
        String question;
        String[] options;
        Map<String, Map<PoliticalParty, Double>> optionWeights;
        
        public Question(String question, String[] options) {
            this.question = question;
            this.options = options;
            this.optionWeights = new HashMap<>();
            initializeWeights();
        }
        
        private void initializeWeights() {
            // Initialize weights for each option and party combination
            for (String option : options) {
                Map<PoliticalParty, Double> weights = new HashMap<>();
                for (PoliticalParty party : PoliticalParty.values()) {
                    weights.put(party, 0.25); // Start with equal probability
                }
                optionWeights.put(option, weights);
            }
        }
    }
    
    // Survey response structure
    static class SurveyResponse {
        List<String> answers;
        PoliticalParty actualParty;
        
        public SurveyResponse(List<String> answers, PoliticalParty actualParty) {
            this.answers = new ArrayList<>(answers);
            this.actualParty = actualParty;
        }
    }
    
    // Naive Bayes Classifier
    static class NaiveBayesClassifier {
        private Map<PoliticalParty, Double> priorProbabilities;
        private Map<String, Map<PoliticalParty, Double>> featureProbabilities;
        private List<SurveyResponse> trainingData;
        private List<Question> questions;
        
        public NaiveBayesClassifier(List<Question> questions) {
            this.questions = questions;
            this.priorProbabilities = new HashMap<>();
            this.featureProbabilities = new HashMap<>();
            this.trainingData = new ArrayList<>();
        }
        
        public void train(List<SurveyResponse> data) {
            this.trainingData = new ArrayList<>(data);
            calculatePriorProbabilities();
            calculateFeatureProbabilities();
            updateQuestionWeights();
        }
        
        private void calculatePriorProbabilities() {
            Map<PoliticalParty, Integer> partyCounts = new HashMap<>();
            
            // Count occurrences of each party
            for (SurveyResponse response : trainingData) {
                partyCounts.put(response.actualParty, 
                    partyCounts.getOrDefault(response.actualParty, 0) + 1);
            }
            
            // Calculate probabilities
            int totalResponses = trainingData.size();
            for (PoliticalParty party : PoliticalParty.values()) {
                double probability = (double) partyCounts.getOrDefault(party, 0) / totalResponses;
                priorProbabilities.put(party, probability);
            }
        }
        
        private void calculateFeatureProbabilities() {
            // For each question and answer combination
            for (int questionIndex = 0; questionIndex < questions.size(); questionIndex++) {
                Question question = questions.get(questionIndex);
                
                for (String option : question.options) {
                    String feature = "Q" + questionIndex + "_" + option;
                    Map<PoliticalParty, Double> probabilities = new HashMap<>();
                    
                    for (PoliticalParty party : PoliticalParty.values()) {
                        int featureCount = 0;
                        int partyCount = 0;
                        
                        for (SurveyResponse response : trainingData) {
                            if (response.actualParty == party) {
                                partyCount++;
                                if (questionIndex < response.answers.size() && 
                                    response.answers.get(questionIndex).equals(option)) {
                                    featureCount++;
                                }
                            }
                        }
                        
                        // Laplace smoothing to avoid zero probabilities
                        double probability = (double) (featureCount + 1) / (partyCount + question.options.length);
                        probabilities.put(party, probability);
                    }
                    
                    featureProbabilities.put(feature, probabilities);
                }
            }
        }
        
        private void updateQuestionWeights() {
            // Update question weights based on training data patterns
            for (int questionIndex = 0; questionIndex < questions.size(); questionIndex++) {
                Question question = questions.get(questionIndex);
                
                for (String option : question.options) {
                    Map<PoliticalParty, Double> weights = new HashMap<>();
                    String feature = "Q" + questionIndex + "_" + option;
                    
                    if (featureProbabilities.containsKey(feature)) {
                        Map<PoliticalParty, Double> probabilities = featureProbabilities.get(feature);
                        
                        for (PoliticalParty party : PoliticalParty.values()) {
                            // Weight is based on how strongly this option correlates with the party
                            double weight = probabilities.get(party);
                            weights.put(party, weight);
                        }
                        
                        question.optionWeights.put(option, weights);
                    }
                }
            }
        }
        
        public Map<PoliticalParty, Double> predict(List<String> answers) {
            Map<PoliticalParty, Double> scores = new HashMap<>();
            
            // Initialize scores with prior probabilities
            for (PoliticalParty party : PoliticalParty.values()) {
                scores.put(party, Math.log(priorProbabilities.getOrDefault(party, 0.001)));
            }
            
            // Add feature probabilities
            for (int i = 0; i < Math.min(answers.size(), questions.size()); i++) {
                String feature = "Q" + i + "_" + answers.get(i);
                
                if (featureProbabilities.containsKey(feature)) {
                    Map<PoliticalParty, Double> probabilities = featureProbabilities.get(feature);
                    
                    for (PoliticalParty party : PoliticalParty.values()) {
                        double currentScore = scores.get(party);
                        double featureProb = probabilities.getOrDefault(party, 0.001);
                        scores.put(party, currentScore + Math.log(featureProb));
                    }
                }
            }
            
            // Convert log probabilities back to probabilities and normalize
            double maxScore = Collections.max(scores.values());
            Map<PoliticalParty, Double> probabilities = new HashMap<>();
            double total = 0.0;
            
            for (PoliticalParty party : PoliticalParty.values()) {
                double prob = Math.exp(scores.get(party) - maxScore);
                probabilities.put(party, prob);
                total += prob;
            }
            
            // Normalize to sum to 1
            for (PoliticalParty party : PoliticalParty.values()) {
                probabilities.put(party, probabilities.get(party) / total);
            }
            
            return probabilities;
        }
        
        public PoliticalParty getBestPrediction(List<String> answers) {
            Map<PoliticalParty, Double> probabilities = predict(answers);
            return probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(PoliticalParty.DEMOCRAT);
        }
    }
    
    // Performance metrics calculator
    static class ModelEvaluator {
        public static Map<String, Double> evaluateModel(NaiveBayesClassifier classifier, 
                                                      List<SurveyResponse> testData) {
            Map<String, Double> metrics = new HashMap<>();
            
            int correct = 0;
            Map<PoliticalParty, Integer> truePositives = new HashMap<>();
            Map<PoliticalParty, Integer> falsePositives = new HashMap<>();
            Map<PoliticalParty, Integer> falseNegatives = new HashMap<>();
            
            // Initialize counters
            for (PoliticalParty party : PoliticalParty.values()) {
                truePositives.put(party, 0);
                falsePositives.put(party, 0);
                falseNegatives.put(party, 0);
            }
            
            // Evaluate predictions
            for (SurveyResponse response : testData) {
                PoliticalParty predicted = classifier.getBestPrediction(response.answers);
                PoliticalParty actual = response.actualParty;
                
                if (predicted == actual) {
                    correct++;
                    truePositives.put(actual, truePositives.get(actual) + 1);
                } else {
                    falsePositives.put(predicted, falsePositives.get(predicted) + 1);
                    falseNegatives.put(actual, falseNegatives.get(actual) + 1);
                }
            }
            
            // Calculate accuracy
            double accuracy = (double) correct / testData.size();
            metrics.put("accuracy", accuracy);
            
            // Calculate precision, recall, and F1 for each party
            double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
            int validParties = 0;
            
            for (PoliticalParty party : PoliticalParty.values()) {
                int tp = truePositives.get(party);
                int fp = falsePositives.get(party);
                int fn = falseNegatives.get(party);
                
                if (tp + fp > 0 || tp + fn > 0) {
                    validParties++;
                    
                    double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
                    double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0;
                    double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
                    
                    totalPrecision += precision;
                    totalRecall += recall;
                    totalF1 += f1;
                    
                    metrics.put("precision_" + party.name(), precision);
                    metrics.put("recall_" + party.name(), recall);
                    metrics.put("f1_" + party.name(), f1);
                }
            }
            
            // Calculate macro averages
            if (validParties > 0) {
                metrics.put("macro_precision", totalPrecision / validParties);
                metrics.put("macro_recall", totalRecall / validParties);
                metrics.put("macro_f1", totalF1 / validParties);
            }
            
            return metrics;
        }
    }
    
    // Main application class
    private static final String CSV_FILE = "training_data.csv";
    private static Scanner scanner = new Scanner(System.in);
    private static List<Question> questions;
    private static NaiveBayesClassifier classifier;
    private static List<SurveyResponse> allData;
    
    public static void main(String[] args) {
        System.out.println("=== Political Party Predictor ===");
        System.out.println("This application uses machine learning to predict political party affiliation.");
        System.out.println();
        
        initializeQuestions();
        classifier = new NaiveBayesClassifier(questions);
        allData = new ArrayList<>();
        
        loadTrainingData();
        
        while (true) {
            System.out.println("\n=== Main Menu ===");
            System.out.println("1. Take Survey (with real-time prediction)");
            System.out.println("2. Train Model");
            System.out.println("3. Evaluate Model");
            System.out.println("4. View Training Data Statistics");
            System.out.println("5. Exit");
            System.out.print("Choose an option: ");
            
            int choice = getIntInput();
            
            switch (choice) {
                case 1:
                    takeSurvey();
                    break;
                case 2:
                    trainModel();
                    break;
                case 3:
                    evaluateModel();
                    break;
                case 4:
                    showStatistics();
                    break;
                case 5:
                    System.out.println("Thank you for using Political Party Predictor!");
                    return;
                default:
                    System.out.println("Invalid option. Please try again.");
            }
        }
    }
    
    private static void initializeQuestions() {
        questions = new ArrayList<>();
        
        questions.add(new Question(
            "What should the government do to help the poor?",
            new String[]{"Expand social welfare programs", "Allow school choice with education vouchers", 
                        "Implement job-training programs", "Reduce government intervention"}
        ));
        
        questions.add(new Question(
            "What is your stance on healthcare?",
            new String[]{"Universal healthcare for all", "Market-based healthcare solutions", 
                        "Minimal government healthcare role", "Single-payer system"}
        ));
        
        questions.add(new Question(
            "How should the government handle climate change?",
            new String[]{"Strict environmental regulations", "Market incentives for clean energy", 
                        "Minimal government involvement", "Aggressive climate action and regulation"}
        ));
        
        questions.add(new Question(
            "What is your view on taxation?",
            new String[]{"Progressive taxation on wealthy", "Flat tax for everyone", 
                        "Minimal taxation overall", "High taxes to fund social programs"}
        ));
        
        questions.add(new Question(
            "How should immigration be handled?",
            new String[]{"Path to citizenship for undocumented", "Merit-based immigration system", 
                        "Open borders policy", "Humanitarian focus on refugees"}
        ));
        
        questions.add(new Question(
            "What is your stance on gun control?",
            new String[]{"Strict gun control laws", "Background checks but protect 2nd Amendment", 
                        "Minimal gun restrictions", "Ban assault weapons"}
        ));
        
        questions.add(new Question(
            "How should the economy be managed?",
            new String[]{"Government regulation of markets", "Free market capitalism", 
                        "Minimal government economic role", "Sustainable economic policies"}
        ));
        
        questions.add(new Question(
            "What is your view on social issues?",
            new String[]{"Progressive social policies", "Traditional family values", 
                        "Government stay out of social issues", "Focus on social justice"}
        ));
        
        questions.add(new Question(
            "How should education be funded?",
            new String[]{"Increase public education funding", "School vouchers and choice", 
                        "Privatize education system", "Free public education for all"}
        ));
        
        // Set initial weights based on typical party positions
        setInitialWeights();
    }
    
    private static void setInitialWeights() {
        // Question 1: What should the government do to help the poor?
        Question q1 = questions.get(0);
        q1.optionWeights.get("Expand social welfare programs").put(PoliticalParty.DEMOCRAT, 0.7);
        q1.optionWeights.get("Expand social welfare programs").put(PoliticalParty.GREEN, 0.8);
        q1.optionWeights.get("Allow school choice with education vouchers").put(PoliticalParty.REPUBLICAN, 0.6);
        q1.optionWeights.get("Reduce government intervention").put(PoliticalParty.LIBERTARIAN, 0.8);
        
        // Question 2: Healthcare stance
        Question q2 = questions.get(1);
        q2.optionWeights.get("Universal healthcare for all").put(PoliticalParty.DEMOCRAT, 0.8);
        q2.optionWeights.get("Market-based healthcare solutions").put(PoliticalParty.REPUBLICAN, 0.7);
        q2.optionWeights.get("Minimal government healthcare role").put(PoliticalParty.LIBERTARIAN, 0.8);
        q2.optionWeights.get("Single-payer system").put(PoliticalParty.GREEN, 0.9);
        
        // Question 3: Climate change
        Question q3 = questions.get(2);
        q3.optionWeights.get("Aggressive climate action and regulation").put(PoliticalParty.GREEN, 0.9);
        q3.optionWeights.get("Strict environmental regulations").put(PoliticalParty.DEMOCRAT, 0.7);
        q3.optionWeights.get("Market incentives for clean energy").put(PoliticalParty.REPUBLICAN, 0.6);
        q3.optionWeights.get("Minimal government involvement").put(PoliticalParty.LIBERTARIAN, 0.8);
        
        // Continue setting weights for other questions...
        // (Similar pattern for remaining questions)
    }
    
    private static void takeSurvey() {
        System.out.println("\n=== Political Ideology Survey ===");
        System.out.println("Answer the following questions. We'll predict your political party as you go!");
        System.out.println();
        
        List<String> answers = new ArrayList<>();
        
        for (int i = 0; i < questions.size(); i++) {
            Question question = questions.get(i);
            System.out.println("Question " + (i + 1) + ": " + question.question);
            
            for (int j = 0; j < question.options.length; j++) {
                System.out.println((char)('A' + j) + ". " + question.options[j]);
            }
            
            System.out.print("Your answer (A-" + (char)('A' + question.options.length - 1) + "): ");
            String input = scanner.nextLine().trim().toUpperCase();
            
            while (input.length() != 1 || input.charAt(0) < 'A' || 
                   input.charAt(0) >= 'A' + question.options.length) {
                System.out.print("Invalid input. Please enter A-" + 
                    (char)('A' + question.options.length - 1) + ": ");
                input = scanner.nextLine().trim().toUpperCase();
            }
            
            int optionIndex = input.charAt(0) - 'A';
            answers.add(question.options[optionIndex]);
            
            // Show real-time prediction if we have training data
            if (!allData.isEmpty() && i >= 2) { // Start predicting after 3 questions
                Map<PoliticalParty, Double> predictions = classifier.predict(answers);
                System.out.println("\n--- Current Prediction ---");
                predictions.entrySet().stream()
                    .sorted(Map.Entry.<PoliticalParty, Double>comparingByValue().reversed())
                    .forEach(entry -> 
                        System.out.printf("%s: %.1f%%\n", 
                            entry.getKey().name(), entry.getValue() * 100));
                System.out.println();
            }
        }
        
        // Final prediction
        if (!allData.isEmpty()) {
            Map<PoliticalParty, Double> finalPredictions = classifier.predict(answers);
            PoliticalParty bestPrediction = classifier.getBestPrediction(answers);
            
            System.out.println("\n=== Final Prediction ===");
            finalPredictions.entrySet().stream()
                .sorted(Map.Entry.<PoliticalParty, Double>comparingByValue().reversed())
                .forEach(entry -> 
                    System.out.printf("%s: %.1f%%\n", 
                        entry.getKey().name(), entry.getValue() * 100));
            
            System.out.println("\nOur best guess: " + bestPrediction.name());
        }
        
        // Ask for actual party affiliation to add to training data
        System.out.println("\nTo help improve our model, please tell us your actual political party:");
        System.out.println("1. Democrat");
        System.out.println("2. Republican"); 
        System.out.println("3. Libertarian");
        System.out.println("4. Green");
        System.out.println("5. Skip (don't save response)");
        System.out.print("Your choice: ");
        
        int partyChoice = getIntInput();
        
        if (partyChoice >= 1 && partyChoice <= 4) {
            PoliticalParty actualParty = PoliticalParty.values()[partyChoice - 1];
            SurveyResponse response = new SurveyResponse(answers, actualParty);
            allData.add(response);
            saveTrainingData();
            
            System.out.println("Thank you! Your response has been saved to improve the model.");
            
            if (!allData.isEmpty()) {
                PoliticalParty predicted = classifier.getBestPrediction(answers);
                if (predicted == actualParty) {
                    System.out.println("Great! Our prediction was correct!");
                } else {
                    System.out.println("We got it wrong this time, but this helps us learn!");
                }
            }
        }
    }
    
    private static void trainModel() {
        if (allData.isEmpty()) {
            System.out.println("No training data available. Please take some surveys first.");
            return;
        }
        
        System.out.println("\n=== Training Model ===");
        System.out.println("Training data size: " + allData.size());
        
        classifier.train(allData);
        
        System.out.println("Model training completed!");
        
        // Show learned probabilities
        System.out.println("\nPrior probabilities:");
        for (PoliticalParty party : PoliticalParty.values()) {
            double prior = classifier.priorProbabilities.getOrDefault(party, 0.0);
            System.out.printf("%s: %.3f\n", party.name(), prior);
        }
    }
    
    private static void evaluateModel() {
        if (allData.size() < 10) {
            System.out.println("Need at least 10 data points for evaluation. Current size: " + allData.size());
            return;
        }
        
        System.out.println("\n=== Model Evaluation ===");
        
        // Split data into training (80%) and testing (20%)
        Collections.shuffle(allData);
        int splitIndex = (int) (allData.size() * 0.8);
        
        List<SurveyResponse> trainingData = allData.subList(0, splitIndex);
        List<SurveyResponse> testData = allData.subList(splitIndex, allData.size());
        
        System.out.println("Training set size: " + trainingData.size());
        System.out.println("Test set size: " + testData.size());
        
        // Train on training data
        NaiveBayesClassifier evalClassifier = new NaiveBayesClassifier(questions);
        evalClassifier.train(trainingData);
        
        // Evaluate on test data
        Map<String, Double> metrics = ModelEvaluator.evaluateModel(evalClassifier, testData);
        
        System.out.println("\n=== Performance Metrics ===");
        System.out.printf("Accuracy: %.3f\n", metrics.get("accuracy"));
        System.out.printf("Macro Precision: %.3f\n", metrics.getOrDefault("macro_precision", 0.0));
        System.out.printf("Macro Recall: %.3f\n", metrics.getOrDefault("macro_recall", 0.0));
        System.out.printf("Macro F1-Score: %.3f\n", metrics.getOrDefault("macro_f1", 0.0));
        
        System.out.println("\n=== Per-Party Metrics ===");
        for (PoliticalParty party : PoliticalParty.values()) {
            String partyName = party.name();
            Double precision = metrics.get("precision_" + partyName);
            Double recall = metrics.get("recall_" + partyName);
            Double f1 = metrics.get("f1_" + partyName);
            
            if (precision != null) {
                System.out.printf("%s - Precision: %.3f, Recall: %.3f, F1: %.3f\n",
                    partyName, precision, recall, f1);
            }
        }
    }
    
    private static void showStatistics() {
        System.out.println("\n=== Training Data Statistics ===");
        System.out.println("Total responses: " + allData.size());
        
        if (allData.isEmpty()) {
            System.out.println("No data available.");
            return;
        }
        
        // Count by party
        Map<PoliticalParty, Integer> partyCounts = new HashMap<>();
        for (SurveyResponse response : allData) {
            partyCounts.put(response.actualParty, 
                partyCounts.getOrDefault(response.actualParty, 0) + 1);
        }
        
        System.out.println("\nResponses by party:");
        for (PoliticalParty party : PoliticalParty.values()) {
            int count = partyCounts.getOrDefault(party, 0);
            double percentage = allData.isEmpty() ? 0 : (double) count / allData.size() * 100;
            System.out.printf("%s: %d (%.1f%%)\n", party.name(), count, percentage);
        }
        
        // Show question response patterns
        System.out.println("\nMost common responses by question:");
        for (int i = 0; i < questions.size(); i++) {
            Map<String, Integer> answerCounts = new HashMap<>();
            
            for (SurveyResponse response : allData) {
                if (i < response.answers.size()) {
                    String answer = response.answers.get(i);
                    answerCounts.put(answer, answerCounts.getOrDefault(answer, 0) + 1);
                }
            }
            
            String mostCommon = answerCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("N/A");
            
            System.out.printf("Q%d: %s\n", i + 1, mostCommon);
        }
    }
    
    private static void loadTrainingData() {
        try {
            File file = new File(CSV_FILE);
            if (!file.exists()) {
                System.out.println("No existing training data found. Starting fresh.");
                return;
            }
            
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line = reader.readLine(); // Skip header
            
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
                
                if (parts.length >= questions.size() + 1) {
                    List<String> answers = new ArrayList<>();
                    for (int i = 0; i < questions.size(); i++) {
                        answers.add(parts[i].replace("\"", ""));
                    }
                    
                    try {
                        PoliticalParty party = PoliticalParty.valueOf(parts[questions.size()]);
                        allData.add(new SurveyResponse(answers, party));
                    } catch (IllegalArgumentException e) {
                        System.out.println("Warning: Invalid party in data: " + parts[questions.size()]);
                    }
                }
            }
            reader.close();
            
            System.out.println("Loaded " + allData.size() + " responses from training data.");
            
            if (!allData.isEmpty()) {
                classifier.train(allData);
                System.out.println("Model trained with existing data.");
            }
            
        } catch (IOException e) {
            System.out.println("Error loading training data: " + e.getMessage());
        }
    }
    
    private static void saveTrainingData() {
        try {
            PrintWriter writer = new PrintWriter(new FileWriter(CSV_FILE));
            
            // Write header
            for (int i = 0; i < questions.size(); i++) {
                writer.print("Question" + (i + 1) + ",");
            }
            writer.println("Party");
            
            // Write data
            for (SurveyResponse response : allData) {
                for (String answer : response.answers) {
                    writer.print("\"" + answer + "\",");
                }
                writer.println(response.actualParty.name());
            }
            
            writer.close();
        } catch (IOException e) {
            System.out.println("Error saving training data: " + e.getMessage());
        }
    }
    
    private static int getIntInput() {
        while (true) {
            try {
                String input = scanner.nextLine().trim();
                return Integer.parseInt(input);
            } catch (NumberFormatException e) {
                System.out.print("Please enter a valid number: ");
            }
        }
    }
}
