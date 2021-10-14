/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>
#include <vector>

class agent
{
public:
	agent(const std::string &args = "")
	{
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair;)
		{
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = {value};
		}
		for (std::map<key, value>::const_iterator it = meta.begin();
			 it != meta.end(); ++it)
		{
			std::cout << it->first << "=" << it->second.value << ";";
		}
		std::cout << std::endl;
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string &flag = "") {}
	virtual void close_episode(const std::string &flag = "") {}
	virtual action take_action(const board &b) { return action(); }
	virtual bool check_for_win(const board &b) { return false; }

public:
	virtual std::string property(const std::string &key) const { return meta.at(key); }
	virtual void notify(const std::string &msg) { meta[msg.substr(0, msg.find('='))] = {msg.substr(msg.find('=') + 1)}; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value
	{
		std::string value;
		operator std::string() const { return value; }
		template <typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent
{
public:
	random_agent(const std::string &args = "") : agent(args)
	{
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class player : public random_agent
{

public:
	player(const std::string &args = "") : random_agent("name=TD alpha=0.005 role=player " + args), alpha(0), opcode({0, 1, 2, 3})
	{
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~player()
	{
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

	virtual void open_episode(const std::string &flag = "")
	{
		history.clear();
	}

	virtual void close_episode(const std::string &flag = "")
	{
		if (history.empty())
			return;
		if (alpha == 0)
			return;
		adjust_value(history[history.size() - 1].after, 0);
		for (int i = history.size() - 2; i >= 0; i--)
		{
			adjust_value(history[i].after, history[i + 1].reward + estimate_value(history[i + 1].after));
		}
	}

protected:
	// TD / n-tuple
	struct step
	{
		int reward;
		board after;
	};
	std::vector<step> history;
	static const int indexCount = 17;
	static const int tupleSize = 4;
	static const int maxIndex = 25;
	int indexes[indexCount][tupleSize] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}, {0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}, {0, 1, 4, 5}, {1, 2, 5, 6}, {2, 3, 6, 7}, {4, 5, 8, 9}, {4, 5, 9, 10}, {4, 5, 10, 11}, {8, 9, 12, 13}, {9, 10, 13, 14}, {10, 11, 14, 15}};

	static float pow(int n, int p)
	{
		float base = 1;

		for (int i = 0; i < p; i++)
			base *= n;
		return base;
	}
	int extract_feature(const board &after, int a)
	{
		int result = 0;
		for (int i = 0; i < tupleSize; i++)
		{
			int tmp = after(indexes[a][i]);
			if (tmp >= maxIndex)
				tmp = maxIndex - 1;
			result += tmp * pow(maxIndex, tupleSize - i - 1);
		}
		return result;
	}

	int extract_feature(const board &after, int a, int b, int c, int d)
	{
		int result = 0;
		for (int i = 0; i < tupleSize; i++)
		{
			int tmp = after(indexes[a][i]);
			if (tmp >= maxIndex)
				tmp = maxIndex - 1;
			result += tmp * pow(maxIndex, tupleSize - i - 1);
		}
		return result;
	}

	int extract_feature(const board &after, int a, int b, int c, int d, int e, int f)
	{
		int result = 0;
		for (int i = 0; i < tupleSize; i++)
		{
			int tmp = after(indexes[a][i]);
			if (tmp >= maxIndex)
				tmp = maxIndex - 1;
			result += tmp * pow(maxIndex, tupleSize - i - 1);
		}
		return result;
	}

	float estimate_value(const board &after)
	{
		float value = 0;
		for (int x = 0; x < indexCount; x++)
			value += net[x][extract_feature(after, x)];

		return value;
	}

	void adjust_value(const board &after, float target)
	{
		float current = estimate_value(after);
		float error = target - current;
		float adjust = alpha * error;
		for (int x = 0; x < indexCount; x++)
			net[x][extract_feature(after, x)] += adjust;
	}

	action td_nTuple_action(const board &before)
	{
		int best_op = -1;
		int best_reward = -1;
		float best_value = -100000;
		board best_afterstate;
		for (int op : opcode)
		{
			board after = before;
			int reward = after.slide(op);
			if (reward == -1)
				continue;
			float value = estimate_value(after);
			if (reward + value > best_reward + best_value)
			{
				best_op = op;
				best_reward = reward;
				best_value = value;
				best_afterstate = after;
			}
		}
		if (best_op != -1)
			history.push_back({best_reward, best_afterstate});
		else
			return action();
		return action::slide(best_op);
	}

	// baseline models
	action dummy_action(const board &before)
	{
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode)
		{
			board::reward reward = board(before).slide(op);
			if (reward != -1)
				return action::slide(op);
		}
		return action();
	}

	action greedy_score_action(const board &before)
	{
		board::reward best_reward = -1;
		int best_op;
		for (int op : opcode)
		{
			board::reward reward = board(before).slide(op);
			if (reward > best_reward)
			{
				best_op = op;
				best_reward = reward;
			}
		}
		if (best_reward != -1)
			return action::slide(best_op);
		return action();
	}

	action greedy_pos_action(const board &before)
	{
		board::reward best_reward = -1;
		unsigned best_space = 17;
		int best_op;
		for (int op : opcode)
		{
			board tmp(before);
			board::reward reward = tmp.slide(op);
			if (reward == -1)
				continue;
			unsigned space_left = tmp.space_left();
			if (reward > best_reward || (reward == best_reward && space_left < best_space))
			{
				best_op = op;
				best_reward = reward;
				best_space = space_left;
			}
		}
		if (best_reward != -1)
			return action::slide(best_op);
		return action();
	}

	virtual action take_action(const board &before)
	{
		if (property("name") == "greedy_score")
			return greedy_score_action(before);
		else if (property("name") == "greedy_pos")
			return greedy_pos_action(before);
		else if (property("name") == "TD")
			return td_nTuple_action(before);
		else if (property("name") == "dummy")
			return dummy_action(before);
		else
			throw std::invalid_argument(property("name") + " is not a valid player name");
	}

	virtual void init_weights(const std::string &info)
	{
		for (int i = 0; i < indexCount; i++)
			net.emplace_back(pow(maxIndex, tupleSize));
	}
	virtual void load_weights(const std::string &path)
	{
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open())
			std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char *>(&size), sizeof(size));
		net.resize(size);
		for (weight &w : net)
			in >> w;
		in.close();
	}
	virtual void save_weights(const std::string &path)
	{
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open())
			std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char *>(&size), sizeof(size));
		for (weight &w : net)
			out << w;
		out.close();
	}

private:
	float alpha;
	std::array<int, 4> opcode;
	std::vector<weight> net;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent
{
public:
	rndenv(const std::string &args = "") : random_agent("name=random role=environment " + args),
										   space({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), popup(0, 9) {}

	virtual action take_action(const board &after)
	{
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space)
		{
			if (after(pos) != 0)
				continue;
			board::cell tile = popup(engine) ? 1 : 2;
			return action::place(pos, tile);
		}
		return action();
	}

private:
	std::array<int, 16> space;
	std::uniform_int_distribution<int> popup;
};
