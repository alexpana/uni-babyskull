#pragma once
#include <queue>
#include <mutex>

template<typename T>
class synchronized_queue //Non copyable!. And there is a good reason for that
{
	std::queue<T> _queue;
	std::mutex _mtx;

public:
	void push(T const &  element) {
		std::lock_guard<std::mutex> lock(_mtx);
		_queue.push(element);
	}

	void pop() {
		std::lock_guard<std::mutex> lock(_mtx);
		_queue.pop();		
	}

	T front() {
		std::lock_guard<std::mutex> lock(_mtx);
		return _queue.front();		
	}

	T back() {
		std::lock_guard<std::mutex> lock(_mtx);
		return _queue.back();
	}

	bool empty() {
		std::lock_guard<std::mutex> lock(_mtx);
		return _queue.empty();
	}
};