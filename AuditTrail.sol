// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuditTrail {
    struct ModelUpdate {
        uint round;
        string ipfsHash;
        uint timestamp;
        address author;
    }

    ModelUpdate[] public history;
    mapping(uint => ModelUpdate) public updatesByRound;

    event ModelUpdated(
        uint indexed round,
        string ipfsHash,
        address indexed author
    );

    function recordModel(uint _round, string memory _ipfsHash) public {
        ModelUpdate memory newUpdate = ModelUpdate({
            round: _round,
            ipfsHash: _ipfsHash,
            timestamp: block.timestamp,
            author: msg.sender
        });

        history.push(newUpdate);
        updatesByRound[_round] = newUpdate;

        emit ModelUpdated(_round, _ipfsHash, msg.sender);
    }

    function getHistoryCount() public view returns (uint) {
        return history.length;
    }
}
