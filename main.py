from src.evolutionary import evolution
from src.visuals import visualise_evolution

if __name__ == "__main__":
    
    config = {
        "N_initial_population": 100,
        "N_iterations": 10,
        "mutation_coefficient": 0.1,
        "fertility_rate": 3,
    }
    x,y, best_coefs = evolution(**config)

    visualise_evolution(x,y, best_coefs)


# TODO:
# - Initialise git repo x
# - Add subfolders and modularise code x
# - Doc strings and type annotations x
# - Add possibility to input any polynomial
# - Add possibility for user to input polynomial in dash
# - Profile code
# - Optimise code
# - Add precommit hooks
# - Add into CV/github.io
# - Deploy
# - Add github workflows (Auto deploy when new stuff)


