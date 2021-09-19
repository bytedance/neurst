# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
try:
    from simuleval import READ_ACTION, WRITE_ACTION, options
    from simuleval.cli import DataWriter, server
    from simuleval.online import start_client, start_server
    from simuleval.utils.agent_finder import find_agent_cls
    from simuleval.utils.functional import split_list_into_chunks
except ModuleNotFoundError:
    pass

import importlib
import json
import logging
import os
import sys
import time
from functools import partial
from multiprocessing import Manager, Pool, Process

from neurst.utils.registry import get_registered_class

logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger('simuleval.cli')


# added here
def init():
    global tf
    import tensorflow as tf
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


def evaluate(args, client, server_process=None):
    info = client.corpus_info()
    num_sentences = info['num_sentences']
    indices = list(range(num_sentences))
    num_processes = args.num_processes
    manager = Manager()
    result_queue = manager.Queue()
    data_writer = DataWriter(args, result_queue)

    if num_processes > 1:
        if num_processes > num_sentences:
            logger.warn(
                f"Number of processes is larger than number sentences ({num_processes}, {num_sentences})."
                f"Will only use {num_sentences} processes"
            )
            num_processes = num_sentences

        # Multi process, split test set into num_processes pieces
        # added here
        with Pool(args.num_processes, initializer=init) as p:
            p.map(
                partial(decode, args, client, result_queue),
                split_list_into_chunks(indices, num_processes),
            )
    else:
        decode(args, client, result_queue, indices)

    scores = client.get_scores()
    logger.info("Evaluation results:\n" + json.dumps(scores, indent=4))
    logger.info("Evaluation finished")

    data_writer.write_scores(scores)
    data_writer.kill()

    if server_process is not None:
        server_process.kill()
        logger.info("Shutdown server")


def decode(args, client, result_queue, instance_ids):
    # Find agent and load related arguments
    if os.path.exists(args.agent):
        agent_name, agent_cls = find_agent_cls(args)
    else:
        agent_cls = get_registered_class(args.agent, "simuleval_agent")
        agent_name = agent_cls.__name__
    logger.info(
        f"Evaluating {agent_name} (process id {os.getpid()}) "
        f"on instances from {instance_ids[0]} to {instance_ids[-1]}"
    )

    parser = options.general_parser()
    options.add_agent_args(parser, agent_cls)
    args, _ = parser.parse_known_args()

    # Data type check
    info = client.corpus_info()
    data_type = info['data_type']
    if data_type != agent_cls.data_type:
        logger.error(
            f"Data type mismatch 'server.data_type {data_type}', "
            f"'{args.agent_cls}.data_type: {args.agent_cls.data_type}'")
        sys.exit(1)

    # build agents
    agent = agent_cls(args)

    # Decode
    for instance_id in instance_ids:
        states = agent.build_states(args, client, instance_id)
        while not states.finish_hypo():
            action = agent.policy(states)
            if action == READ_ACTION:
                states.update_source()
            elif action == WRITE_ACTION:
                prediction = agent.predict(states)
                states.update_target(prediction)
            else:
                raise SystemExit(f"Undefined action name {action}")
        sent_info = client.get_scores(instance_id)
        result_queue.put(sent_info)
        logger.debug(f"Instance {instance_id} finished, results:\n{json.dumps(sent_info, indent=4)}")


def _main(client_only=False):
    parser = options.general_parser()
    options.add_server_args(parser)

    if not client_only:
        options.add_data_args(parser)

    args, _ = parser.parse_known_args()

    if not client_only:
        if os.path.exists(args.agent):
            _, agent_cls = find_agent_cls(args)
        else:
            agent_cls = get_registered_class(args.agent, "simuleval_agent")
        if args.data_type is None:
            args.data_type = agent_cls.data_type
        logging.getLogger("tornado.access").setLevel(logging.WARNING)
        server_process = Process(
            target=start_server, args=(args,))
        server_process.start()
        time.sleep(3)
    else:
        server_process = None

    client = start_client(args)
    evaluate(args, client, server_process)


if __name__ == "__main__":
    try:
        import simuleval

        importlib.import_module("neurst.utils.simuleval_agents")
        _ = simuleval
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install simuleval via: \n"
            "\tgit clone https://github.com/facebookresearch/SimulEval.git\n"
            "\tpip3 install -e SimulEval/")
    parser = options.general_parser()
    options.add_server_args(parser)
    args, _ = parser.parse_known_args()

    if not args.server_only:
        _main(args.client_only)
    else:
        server()
