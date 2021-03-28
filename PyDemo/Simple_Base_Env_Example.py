from mlagents_envs.environment import UnityEnvironment
from mlagents_envs import base_env
import numpy as np 

# UnityEnvironment is the main interface between the Unity application
# and your Python code
env = UnityEnvironment(file_name="PATH TO UNITY BINARY", seed=1, side_channels=[])

env.reset()

# behavior_name is a string that identifies a behavior in the simulation
behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")

# behavior_specs is a describes the shape of the observation data inside 
# DecisionSteps and TerminalSteps as well as the expected action shapes.
# We can use behavior_name to get the shape of the observation data
# and actions for this particular behavior 
spec = env.behavior_specs[behavior_name]



#Here we have demo function which accepts obervations as input and returns 
#actions as output. Expected form of the input is a array while the output
#is as actionTuple
def GetNextAction(observations):
    
    print(observations)
    
    #Create a 2d array of uniform random values in [-1,1] 
    continuousActionVector = 2*np.random.sample(2)-1 #Get random vector 
    continuousActionVector = continuousActionVector.reshape(1,2) #Reshape it
   
    #In this example we are not using a discrete action so 
    discreteActionVector = np.array([]).reshape(1,0);
   
    #Create an ActionTuple
    actionTuple = base_env.ActionTuple(continuousActionVector,discreteActionVector)
    
    return actionTuple

 

for episode in range(20):
  env.reset()
  
  # decision_steps contains the data from Agents belonging to the same 
  #"Behavior" in the simulation, such as observations and rewards.
  #terminal_steps is like decision_steps but only Agents whose episode ended 
  #since the last call to env.step() are in the TerminalSteps object.
  decision_steps, terminal_steps = env.get_steps(behavior_name)
  
  tracked_agent = -1 # -1 indicates not yet tracking
  done = False # For the tracked_agent
  episode_rewards = 0 # For the tracked_agent
  while not done:

    if tracked_agent == -1 and len(decision_steps) >= 1:
        
      #agent_id is an int vector of length batch size containing unique 
      # identifier for the corresponding Agent. This is used to track Agents 
      # across simulation steps. 
      #The batch size is number of agents requesting a decision since the 
      # last call to env.step()
      tracked_agent = decision_steps.agent_id[0] # Track the first agent
      
    
    # Generate an action
    actionTuple = GetNextAction(decision_steps[0].obs[0])

    # Set the actions
    env.set_actions(behavior_name, actionTuple)
    
    # Send a signal to step the simulation forward by an Update call.
    env.step()
    
    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    if tracked_agent in decision_steps: # The agent requested a decision
      #Accumlagted rewards since last call to env.step()
      episode_rewards += decision_steps[tracked_agent].reward 
    if tracked_agent in terminal_steps: # The agent terminated its episode
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
  print(f"Total rewards for episode {episode} is {episode_rewards}")



env.close()
print("Closed environment")

