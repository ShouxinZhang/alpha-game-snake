use snake_alphazero::{InferenceEngine, SelfPlayWorker, SelfPlayConfig, MCTSConfig, GameStats};
use std::sync::Arc;
use std::fs::File;
use std::io::Write;

fn main() {
    // Configuration
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/latest_model.onnx".to_string());
    
    let output_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "data/self_play".to_string());
    
    let num_games: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    
    let grid_size: i32 = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    
    println!("Snake AlphaZero Self-Play Generator");
    println!("====================================");
    println!("Model: {}", model_path);
    println!("Output: {}", output_dir);
    println!("Games: {}", num_games);
    println!("Grid: {}x{}", grid_size, grid_size);
    
    // Initialize inference engine
    println!("\nLoading model...");
    let inference_engine = Arc::new(InferenceEngine::new(&model_path));
    
    // Self-play configuration
    let config = SelfPlayConfig {
        grid_width: grid_size,
        grid_height: grid_size,
        mcts_config: MCTSConfig {
            num_threads: 4, // Reduce for smaller grid
            num_simulations: 50, // Fewer simulations for 5x5
            cpuct: 1.5,
            virtual_loss: 3.0,
        },
        temperature: 1.0,
        temperature_threshold: 10,
    };
    
    let worker = SelfPlayWorker::new(config, inference_engine);
    
    // Create output directory
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    
    // Generate games with statistics
    println!("\nGenerating {} games...\n", num_games);
    println!("{:>6} | {:>8} | {:>8} | {:>8} | {:>8}", "Games", "Avg Steps", "Avg Score", "Avg Reward", "Win Rate");
    println!("{}", "-".repeat(55));
    
    let mut total_samples = 0;
    let mut batch_stats: Vec<GameStats> = Vec::new();
    
    for game_idx in 0..num_games {
        let seed = game_idx + 42;
        let (samples, stats) = worker.play_game(seed);
        
        total_samples += samples.len();
        batch_stats.push(stats);
        
        // Save samples to JSON
        let filename = format!("{}/game_{:05}.json", output_dir, game_idx);
        let json = serde_json::to_string(&samples).expect("Failed to serialize samples");
        let mut file = File::create(&filename).expect("Failed to create file");
        file.write_all(json.as_bytes()).expect("Failed to write file");
        
        // Print stats every 10 games
        if (game_idx + 1) % 10 == 0 {
            let avg_steps: f64 = batch_stats.iter().map(|s| s.steps as f64).sum::<f64>() / batch_stats.len() as f64;
            let avg_score: f64 = batch_stats.iter().map(|s| s.score as f64).sum::<f64>() / batch_stats.len() as f64;
            let avg_reward: f64 = batch_stats.iter().map(|s| s.final_reward as f64).sum::<f64>() / batch_stats.len() as f64;
            let win_rate: f64 = batch_stats.iter().filter(|s| s.is_victory).count() as f64 / batch_stats.len() as f64 * 100.0;
            
            println!("{:>6} | {:>8.2} | {:>8.2} | {:>8.3} | {:>7.1}%", 
                game_idx + 1, avg_steps, avg_score, avg_reward, win_rate);
            
            batch_stats.clear();
        }
    }
    
    println!("\n====================================");
    println!("Done! Generated {} samples from {} games.", total_samples, num_games);
    println!("Output saved to: {}", output_dir);
}
