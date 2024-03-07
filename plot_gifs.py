from execution import *
import imageio.v2

precision = 'double'

def create_gif(num_threads, dt, method, cell_model, dx, mode):

    times = []
    frame = []
    frames = []

    frames_file = f'./simulation-files/{precision}/{mode}/{dx}/{cell_model}/{method}/frames-{num_threads}-{dt}.txt'
    f = open(frames_file, 'r')
    
    line = f.readline()
    line = line.split()
    frame_count = 0
    
    while True:
        if not line:
            break
        if len(line) == 1:
            times.append(float(line[0]))
            frame = []
            line = f.readline()
            if not line:
                break
            line = line.split()
        else:
            while len(line) > 1:
                # line = line.split()
                frame.append(line)
                line = f.readline()
                if not line:
                    break
                line = line.split()
                
            frame_name = f'frame-{frame_count}.png'
            frames.append(frame_name)
            
            for i in range(len(frame)):
                frame[i] = [float(x) for x in frame[i]]
            
            plt.imshow(frame, cmap='plasma', vmin=0.0, vmax=100)
            plt.colorbar(label='V (mV)')
            plt.title(f'{mode} {cell_model} {method} dt = {dt} t = {times[frame_count]:.2f}')
            plt.xticks([])
            plt.yticks([])
            
            plt.savefig(frame_name)
            plt.close()
            
            frame_count += 1

    # Create gif directory
    if not os.path.exists(f'./gifs/{mode}/{dx}/{cell_model}/{method}'):
        os.makedirs(f'./gifs/{mode}/{dx}/{cell_model}/{method}')

    # Build gif
    with imageio.v2.get_writer(f'./gifs/{mode}/{dx}/{cell_model}/{method}/{num_threads}-{dt}.gif', mode='I') as writer:
        for frame in frames:
            image = imageio.v2.imread(frame)
            writer.append_data(image)

    # Remove files
    for png in set(frames):
        if png.find('lastframe') == -1:
            os.remove(png)

def main():
    for mode in modes:
        for dx in dxs:
            for cell_model in cell_models:
                for number_threads in numbers_threads:
                    for method in methods:
                        for dt in dts:
                            create_gif(number_threads, f'{dt:.3f}', method, cell_model, dx, mode)
                            print(f'Gif created for {cell_model} with {number_threads} threads, {dt} dt, {dx} dx and {method} method ruuning on {mode}')

if __name__ == '__main__':
    main()