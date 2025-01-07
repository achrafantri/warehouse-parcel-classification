package com.example.achats.repository;

import com.example.achats.model.Achat;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface AchatRepository extends JpaRepository<Achat, Long> {
    // Additional query methods can be defined here if needed
}
