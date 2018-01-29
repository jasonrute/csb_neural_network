# Training a Neural Network to play Coders Strike Back #

Here I will describe how I trained a neural network to play the game [Coders Strike Back](https://www.codingame.com/multiplayer/bot-programming/coders-strike-back).  This is a racing game where one of your two bots must complete a race course before the opposing pair of bots.  Here you can see a clip of my bot in action.

![My bot navigating the race course](bot_nn_genetic.gif)

At each time step you can adjust the direction and thrust of your bot.  Because of the collisions and the weird physics, it is not easy to just code a simple policy algorithm for your bot to efficiently complete the course.  All the good players simulate the physics of the course exactly.  Then most of them use some sort of genetic (or Monte Carlo, simulated annealing) algorithm to search for a near optimal policy, using a hand-crafted evaluation function.  The idea is that for each move, they simulate random trajectories, score how they do in the simulation, and modify the best ones slightly, and simulating those.  (See [Magus' great tutorial](http://files.magusgeek.com/csb/csb_en.html) for instructions on how to set up a simulator and genetic algorithm for Coders Strike Back.)

However, especially if programming in Python, it is quite computationally expensive to run a lot of simulations, so the final result of the genetic algorithm is somewhat based on luck.  If you started with a good random solution, then you are more likely to have a good solution in the end.  One way to improve this is to *train* the algorithm to *remember* good moves.  This can be done via a neural network approach.

1. Simulate the game play environment.
2. Construct a genetic algorithm which can find the best(-ish) move.
3. Use data from running that algorithm many times to develop a set of training data.
4. Train a neural network to recognize the best moves.
5. Use that neural network policy alone, or better yet, use that neural network policy as a starting point for the genetic algorithm.
6. Repeat steps 3-6 to continuously improve the neural network policy.

Doing this significantly improved my bot from just a genetic algorithm alone.

## Details ##

**(I will update this section with more details later.)**  
Some quick points:
- I used just a standard feed-forward neural network with dense layers
- I trained the network with TensorFlow on Python, but ran it using custom built neural network code written in Python/NumPy.  (See [here](https://github.com/jasonrute/Neural-Network) for the code.)
- I only trained a racing policy.  There was no blocking strategy, collision avoidance, and any other iterations with other racing bots.  (However, I did add those things to the genetic algorithm used in the actual game play.)
- The inputs for the neural network were the relative positions of the next three checkpoints.
- The outputs where the new acceleration and the relative change in angle.
- Agada used deep Q-learning and got better results.  I think the big difference is that deep Q-learning (and other forms of reinforcement learning) allows the neural network to learn its own evaluation function, were I relied on a custom (and likely imperfect) evaluation function.

# Code #
Since it is a competition, I am not going to give out my code.  

However, you can find an [implementation of a neural network](https://github.com/jasonrute/Neural-Network) written in Python/Numpy that can run inside CodinGames.  It doesn't take up too many lines of code (especially if you remove all the lines related to training it, which you don't need to do inside the competition code.)

When saving the neural network weights, you can use float16 to save space.  Then to store them in your competition code (which has a limit on the file size) do the following:
- convert to bytes with pickle.dumps
- compress with zlib.compress
- convert to ASCII with b64.encode
- paste that ASCII string into the code
- decode with `pickle.loads(zlib.decompress(b64.b64decode(string_repr_of_object)))`
