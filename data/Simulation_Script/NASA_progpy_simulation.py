import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import progpy
from progpy.models import BatteryElectroChem
from progpy.models import BatteryCircuit
from progpy.models import BatteryElectroChemEODEOL

# Initialize the BatteryElectroChem model, which integrates both EOD and EOL models
m = BatteryElectroChem()
print(m.events)
# Define simulation options
options = {
    'save_freq': 10,  # Frequency at which results are saved
    'dt': 2,  # Timestep in seconds # Threshold keys to monitor
}


# Redefine the event_state method for the BatteryElectroChem model
def event_state(x):
    # Constants of nature
    R = 8.3144621  # universal gas constant, J/K/mol
    F = 96487      # Faraday's constant, C/mol
    
    params = m.parameters
    An = params['An']
    Ap = params['Ap']

    # Negative Surface
    xnS = np.clip(x['qnS'] / params['qSMax'], 1e-6, 1 - 1e-6)  # Avoid division by zero or log of zero
    xnS2 = xnS+xnS
    one_minus_xnS = 1 - xnS
    xnS2_minus_1 = xnS2 - 1
    VenParts = [
            An[0] *xnS2_minus_1/F,  # Ven0
            An[1] *(xnS2_minus_1**2  - (xnS2*one_minus_xnS))/F,  # Ven1
            An[2] *(xnS2_minus_1**3  - (4 *xnS*one_minus_xnS)*xnS2_minus_1)/F,  #Ven2
            An[3] *(xnS2_minus_1**4  - (6 *xnS*one_minus_xnS)*xnS2_minus_1**2) /F,  #Ven3
            An[4] *(xnS2_minus_1**5  - (8 *xnS*one_minus_xnS)*xnS2_minus_1**3) /F,  #Ven4
            An[5] *(xnS2_minus_1**6  - (10*xnS*one_minus_xnS)*xnS2_minus_1**4) /F,  #Ven5
            An[6] *(xnS2_minus_1**7  - (12*xnS*one_minus_xnS)*xnS2_minus_1**5) /F,  #Ven6
            An[7] *(xnS2_minus_1**8  - (14*xnS*one_minus_xnS)*xnS2_minus_1**6) /F,  #Ven7
            An[8] *(xnS2_minus_1**9  - (16*xnS*one_minus_xnS)*xnS2_minus_1**7) /F,  #Ven8
            An[9] *(xnS2_minus_1**10 - (18*xnS*one_minus_xnS)*xnS2_minus_1**8) /F,  #Ven9
            An[10]*(xnS2_minus_1**11 - (20*xnS*one_minus_xnS)*xnS2_minus_1**9) /F,  #Ven10
            An[11]*(xnS2_minus_1**12 - (22*xnS*one_minus_xnS)*xnS2_minus_1**10)/F,  #Ven11
            An[12]*(xnS2_minus_1**13 - (24*xnS*one_minus_xnS)*xnS2_minus_1**11)/F   #Ven12
        ]
    Ven = params['U0n'] + R*x['tb']/F*np.log(one_minus_xnS/xnS) + sum(VenParts)


    # Positive Surface
    xpS = np.clip(x['qpS'] / params['qSMax'], 1e-6, 1 - 1e-6)  # Avoid division by zero or log of zero
    one_minus_xpS = 1 - xpS
    xpS2 = xpS + xpS
    xpS2_minus_1 = xpS2 - 1
    VepParts = [
        Ap[0] *(xpS2_minus_1)/F,  #Vep0
        Ap[1] *((xpS2_minus_1)**2  - (xpS2*one_minus_xpS))/F,  #Vep1 
        Ap[2] *((xpS2_minus_1)**3  - (4 *xpS*one_minus_xpS)*(xpS2_minus_1)) /F,  #Vep2
        Ap[3] *((xpS2_minus_1)**4  - (6 *xpS*one_minus_xpS)*(xpS2_minus_1)**(2)) /F,  #Vep3
        Ap[4] *((xpS2_minus_1)**5  - (8 *xpS*one_minus_xpS)*(xpS2_minus_1)**(3)) /F,  #Vep4
        Ap[5] *((xpS2_minus_1)**6  - (10*xpS*one_minus_xpS)*(xpS2_minus_1)**(4)) /F,  #Vep5
        Ap[6] *((xpS2_minus_1)**7  - (12*xpS*one_minus_xpS)*(xpS2_minus_1)**(5)) /F,  #Vep6
        Ap[7] *((xpS2_minus_1)**8  - (14*xpS*one_minus_xpS)*(xpS2_minus_1)**(6)) /F,  #Vep7
        Ap[8] *((xpS2_minus_1)**9  - (16*xpS*one_minus_xpS)*(xpS2_minus_1)**(7)) /F,  #Vep8
        Ap[9] *((xpS2_minus_1)**10 - (18*xpS*one_minus_xpS)*(xpS2_minus_1)**(8)) /F,  #Vep9
        Ap[10]*((xpS2_minus_1)**11 - (20*xpS*one_minus_xpS)*(xpS2_minus_1)**(9)) /F,  #Vep10
        Ap[11]*((xpS2_minus_1)**12 - (22*xpS*one_minus_xpS)*(xpS2_minus_1)**(10))/F,  #Vep11
        Ap[12]*((xpS2_minus_1)**13 - (24*xpS*one_minus_xpS)*(xpS2_minus_1)**(11))/F   #Vep12
    ]
    Vep = params['U0p'] + R*x['tb']/F*np.log(one_minus_xpS/xpS) + sum(VepParts)

    # Battery voltage
    v = Vep - Ven - x['Vo'] - x['Vsn'] - x['Vsp']

    # State of Charge (SOC)
    charge_EOD = (x['qnS'] + x['qnB']) / params['qnMax']
    voltage_EOD = (v - params['VEOD']) / params['VDropoff']

    # End of Life (EOL) State - capacity degradation
    qMax_current = x['qMax']
    qMax_initial = params['x0']['qMax']
    eol_state = (qMax_current - params['qMaxThreshold']) / (qMax_initial - params['qMaxThreshold'])
    return {
        'SOC': charge_EOD,
        'voltage': v,
        'EOD': np.minimum(charge_EOD, voltage_EOD),
        'EOL': np.clip(eol_state, 0, 1),
    }

def run_simulation(discharge_voltage, simulation_dataset_path):
    def future_loading(t, x=None):
        if x is not None:
            event_state = future_loading.event_state(x)
            soc = event_state['SOC']
            voltage = event_state['voltage']
            eol = event_state['EOL']
            # Stop the simulation if the battery reaches a critical degradation state

            if future_loading.mode == 'discharge':
                if voltage <= discharge_voltage:
                    future_loading.mode = 'charge_cc'
                    print(f"Transitioning to CC mode at t={t:.2f}s")
                return m.InputContainer({'i': future_loading.discharge_current})
        
            elif future_loading.mode == 'charge_cc':
                if voltage >= 4.2:
                    future_loading.mode = 'charge_cv'
                    future_loading.start = future_loading.charge_current  # Start with 0 current in CV mode
                    future_loading.cv_start_time = 0
                    print(f"Transitioning to CV mode at t={t:.2f}s")
                else:
                    return m.InputContainer({'i': -future_loading.charge_current})
                
            elif future_loading.mode == 'charge_cv':
                cv_current = max(future_loading.start - future_loading.cv_slope * future_loading.cv_start_time, 0.02)
                future_loading.cv_start_time += 1
                if cv_current <= 0.02:
                    future_loading.mode = 'discharge'
                    print(f"Transitioning to Discharge mode at t={t:.2f}s")
                return m.InputContainer({'i': -cv_current})
        return m.InputContainer({'i': 0})
    # Initialize future_loading parameters
    future_loading.event_state = m.event_state
    future_loading.charge_current = 1.5  # CC charge current in amps
    future_loading.cv_start_time = 0
    future_loading.cv_slope = 0.0008     # CV mode current slope derived from your dataset
    future_loading.discharge_current = 2.0  # Discharge current in amps
    future_loading.start = future_loading.charge_current
    future_loading.mode = 'discharge'  # Start in discharge_initial
    
    # Simulate the battery degradation process
    simulated_results = m.simulate_to_threshold(future_loading, threshold_keys=['InsufficientCapacity'], **options)
    time = simulated_results.times  # Time series data
    states = simulated_results.states  # States (e.g., voltage, temperature)
    inputs = simulated_results.inputs  # Inputs (e.g., current)
    outputs = simulated_results.outputs  # Outputs (e.g., terminal voltage)

    # Collect capacity and qMax from each state
    capacity = [state['qnS'] + state['qnB'] for state in states]
    qMax = [state['qMax'] for state in states]

    # Convert results to a DataFrame
    data = {
        'time': time,
        'voltage': [output['v'] for output in outputs],
        'temperature': [output['t'] for output in outputs],
        'current': [input['i'] for input in inputs],
        'capacity': capacity,
        'qMax': qMax , # Track the degradation of qMax over time
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(simulation_dataset_path, index=False)

# Assign the custom event_state function to the model
m.event_state = event_state

# Run the simulation
run_simulation(discharge_voltage=2.5, simulation_dataset_path='data/SimulatedData/battery_degradation_simulated_data_18.csv')

