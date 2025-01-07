package com.example.achats.repository;

import com.example.achats.model.BonDeCommande;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BonDeCommandeRepository extends JpaRepository<BonDeCommande, Long> {
    // Méthodes personnalisées peuvent être ajoutées ici si nécessaire
}
