import jax
from absl import app
from absl import flags
flags.DEFINE_string('server_addr', '', help='server ip addr')
flags.DEFINE_integer('num_hosts', 1, help='num of hosts' )
flags.DEFINE_integer('host_idx', 0, help='index of current host' )
FLAGS = flags.FLAGS

def main(argv):
    jax.distributed.initialize(FLAGS.server_addr, FLAGS.num_hosts, FLAGS.host_idx)
    print(f"device count:{jax.device_count()}")
    print(f"process index:{jax.process_index()}")
    print(jax.devices())

if __name__ == '__main__':
  app.run(main)